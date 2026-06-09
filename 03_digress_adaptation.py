"""
CausalDiffChem — Step 3: DiGress Adaptation for Causal Conditioning
Skeleton for adapting the DiGress graph diffusion model to accept pathway
conditioning vectors (Δv) from Module B.

This file contains:
  1. The causal conditioning wrapper around DiGress
  2. The three training loss components (L_pathway, L_ADMET, L_causal)
  3. A linearised SCM causal consistency loss (the novel piece)
  4. Setup instructions for cloning and integrating with the real DiGress repo

To actually run training, you need:
  - A GPU (A100 or equivalent)
  - pip install torch torch_geometric
  - git clone https://github.com/cvignac/DiGress
  - ZINC15 + ChEMBL30 graphs processed by 01_mol_pipeline.py
  - Pathway Δv vectors from Module B (GSVA on your expression data)

This file is intentionally runnable for architecture validation (CPU, no data).
"""

import numpy as np
from pathlib import Path


# ── Availability check ──────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not installed. Run: pip install torch torch_geometric --break-system-packages")
    print("Architecture description will be printed instead.\n")


# ============================================================================
# SECTION 1: Architecture Overview (always prints)
# ============================================================================

ARCHITECTURE_DESCRIPTION = """
CausalDiffChem — Diffusion Model Architecture
==============================================

Module C: Graph Transformer Diffusion (CausalDiffModel)
──────────────────────────────────────────────────────

BASE: DiGress (Vignac et al. 2022, arXiv:2209.14734)
  - Discrete denoising diffusion over molecular graphs
  - Forward process: uniform categorical noise over atom/bond types
  - Reverse: graph transformer denoising network
  - T = 1,000 timesteps, trained on 1.2M ZINC15 + ChEMBL30 molecules

CAUSAL CONDITIONING (our addition):
  Input:  Δv ∈ ℝ^128  (pathway state difference: healthy − disease)
  Source: Module B GAT encoder applied to GSVA pathway activity vectors
  
  Integration points:
    1. Cross-attention layers in graph transformer (Δv as key/value)
    2. Timestep embedding concatenated with projected Δv
    3. Global graph-level conditioning via FiLM (Feature-wise Linear Modulation)

TRAINING OBJECTIVE:
  L_total = L_denoise + λ₁·L_pathway + λ₂·L_ADMET + λ₃·L_causal
  λ₁=0.40, λ₂=0.20, λ₃=0.30

  L_denoise: Standard cross-entropy over noised atom/bond types (from DiGress)
  
  L_pathway: MSE between predicted pathway shift and target Δv
    Uses a frozen copy of the Module B encoder as a "pathway critic"
    
  L_ADMET:   Soft penalties for violating drug-likeness constraints
    - logBB < -3: BBB penalty
    - MW > 600: size penalty  
    - QED < 0.3: drug-likeness penalty
    Computed via differentiable property predictors
    
  L_causal:  Novel — causal consistency loss (linearised SCM approximation)
    Approximates P(Y | do(X=x)) using a linearised structural equation model
    W_lin ∈ ℝ^(d×d): learned adjacency (from NOTEARS, frozen during gen. training)
    For a generated molecule with predicted targets T:
      v_int = (I - W_lin)^{-1} · (v_disease + δ_T)
      L_causal = -cos_sim(v_int, v_healthy)  (same as phenotype reversal score R)
    Gradient flows through W_lin^{-1} · δ_T, where δ_T is differentiable
    via Chemprop's binding predictions.

HARDWARE: 4× A100 80GB, ~3-5 days for 150 epochs on 1.2M molecules
OPTIMIZER: AdamW, lr=1e-4, cosine annealing schedule, batch_size=128

HOW TO ADAPT DiGress:
  git clone https://github.com/cvignac/DiGress
  cd DiGress
  # Add CausalConditioningLayer to src/diffusion/graph_transformer.py
  # Modify compute_loss() in src/diffusion/diffusion_model.py
  # See causaldiffchem/modules/conditioning.py (this file)
"""

print(ARCHITECTURE_DESCRIPTION)


if not TORCH_AVAILABLE:
    print("Install PyTorch to instantiate and validate the architecture:")
    print("  pip install torch torch_geometric --break-system-packages")
    exit(0)


# ============================================================================
# SECTION 2: Causal Conditioning Components (requires PyTorch)
# ============================================================================

class CausalConditioningLayer(nn.Module):
    """
    Injects the pathway conditioning vector Δv into a graph transformer layer.
    Implements FiLM (Feature-wise Linear Modulation):
        h_conditioned = γ(Δv) ⊙ h + β(Δv)
    where γ and β are learned linear projections.
    
    Drop this into DiGress's GraphTransformer after each attention block.
    """

    def __init__(self, node_dim: int, cond_dim: int = 128):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, node_dim)
        self.beta  = nn.Linear(cond_dim, node_dim)
        # LayerNorm for stability
        self.norm  = nn.LayerNorm(node_dim)

    def forward(self, h: 'torch.Tensor', cond_vec: 'torch.Tensor') -> 'torch.Tensor':
        """
        h:        [N_atoms, node_dim]  — node features from attention
        cond_vec: [batch, cond_dim]   — pathway Δv, broadcast to all atoms
        """
        # cond_vec is per-graph; broadcast to per-atom
        # In practice, you pass in graph-level repeated cond_vec
        gamma = self.gamma(cond_vec)   # [batch, node_dim]
        beta  = self.beta(cond_vec)    # [batch, node_dim]
        return self.norm(gamma * h + beta)


class LinearisedSCMLoss(nn.Module):
    """
    Causal consistency loss using a linearised structural equation model.
    
    The key insight: NOTEARS produces a weighted adjacency matrix W.
    For a linear SCM, the equilibrium state is:
        v = (I - W)^{-1} · e
    where e is the external input (intervention + baseline noise).
    
    This is differentiable! We can backpropagate through the matrix inverse.
    
    Usage in training:
        1. Freeze W_lin (learned from NOTEARS on expression data)
        2. Predict binding targets for generated molecule → δ_T (intervention)
        3. Compute v_int = (I - W_lin)^{-1} · (v_disease + δ_T)
        4. L_causal = 1 - cosine_similarity(v_int, v_healthy)
    """

    def __init__(self, W_lin: 'torch.Tensor', v_disease: 'torch.Tensor',
                 v_healthy: 'torch.Tensor'):
        """
        W_lin:     [d, d] weighted adjacency from NOTEARS (frozen)
        v_disease: [d]    disease baseline pathway state
        v_healthy: [d]    healthy reference pathway state
        """
        super().__init__()
        d = W_lin.shape[0]
        I = torch.eye(d)
        # Precompute (I - W)^{-1} — this is fixed during generation training
        M_inv = torch.linalg.solve(I - W_lin, I)
        # Register as buffers (not parameters) — frozen
        self.register_buffer('M_inv',    M_inv)
        self.register_buffer('v_disease', v_disease)
        self.register_buffer('v_healthy', v_healthy)

    def forward(self, delta_T: 'torch.Tensor') -> 'torch.Tensor':
        """
        delta_T: [batch, d] — intervention effect of generated molecule
                               (predicted binding → pathway node perturbation)
                               Differentiable via Chemprop predictions.
        Returns: scalar loss (1 - mean cosine similarity)
        """
        # Post-intervention state: v_int = M_inv · (v_disease + δ_T)
        v_int = (self.v_disease + delta_T) @ self.M_inv.T  # [batch, d]

        # Phenotype reversal score R = cos_sim(v_int, v_healthy)
        R = F.cosine_similarity(v_int, self.v_healthy.unsqueeze(0), dim=1)

        # Loss: we want R → 1, so minimise 1 - R
        return (1.0 - R).mean()


class ADMETLoss(nn.Module):
    """
    Differentiable ADMET penalty for generated molecules.
    
    In practice this requires differentiable property predictors.
    Here we show the structure using placeholder predictors.
    In the real implementation, use:
      - Chemprop for binding affinity (pKi)
      - The B3DB logBB RF model (not differentiable — use STE or surrogate)
      - QED from RDKit (not differentiable — use a neural QED surrogate)
    
    For training, the paper uses soft constraint penalties added to the loss.
    """

    def __init__(self, logbb_threshold: float = -1.0,
                 mw_limit: float = 500.0,
                 qed_threshold: float = 0.5):
        super().__init__()
        self.logbb_threshold = logbb_threshold
        self.mw_limit = mw_limit
        self.qed_threshold = qed_threshold

    def forward(self, node_logits: 'torch.Tensor') -> 'torch.Tensor':
        """
        node_logits: [N_atoms, n_atom_types] — soft atom type distribution
        
        NOTE: In the real implementation, you decode SMILES from node_logits
        and run property predictors. Here we show the penalty structure
        with placeholder differentiable proxies.
        """
        # Placeholder: in real training, decode to mol and compute properties
        # Then apply soft ReLU penalties:
        #
        # logbb_pred = logbb_predictor(mol)   # differentiable surrogate
        # bbb_penalty = F.relu(self.logbb_threshold - logbb_pred).mean()
        #
        # mw_pred = mw_predictor(node_logits)  # differentiable from atom types
        # mw_penalty = F.relu(mw_pred - self.mw_limit).mean()
        #
        # qed_pred = qed_predictor(mol)
        # qed_penalty = F.relu(self.qed_threshold - qed_pred).mean()
        #
        # return bbb_penalty + 0.5 * mw_penalty + qed_penalty

        # Stub: returns 0 (replace with real implementation)
        return torch.tensor(0.0, requires_grad=True)


class CausalDiffModelStub(nn.Module):
    """
    Stub CausalDiffModel matching the paper's architecture description.
    
    Real implementation: adapt DiGress by:
      1. Adding CausalConditioningLayer after each GraphTransformer block
      2. Replacing the loss computation with compute_loss() below
      3. Adding delta_T computation via binding predictions
    
    This stub validates the interface and loss structure on CPU.
    """

    def __init__(self, node_dim: int = 9, edge_dim: int = 4,
                 cond_dim: int = 128, T: int = 1000,
                 scm_dim: int = 64):
        super().__init__()
        self.T        = T
        self.node_dim = node_dim
        self.cond_dim = cond_dim

        # Conditioning injection
        self.cond_proj = nn.Linear(cond_dim, 256)
        self.cond_layer = CausalConditioningLayer(256, cond_dim)

        # Placeholder encoder (replace with DiGress graph transformer)
        self.encoder    = nn.Sequential(
            nn.Linear(node_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),
        )
        self.node_head  = nn.Linear(256, node_dim)
        self.edge_head  = nn.Linear(256, edge_dim)

        # Loss components
        self.admet_loss  = ADMETLoss()

        # Fake SCM buffers for testing (replace with real NOTEARS output)
        W_lin    = torch.zeros(scm_dim, scm_dim)
        v_dis    = torch.randn(scm_dim)
        v_heal   = torch.randn(scm_dim)
        self.causal_loss = LinearisedSCMLoss(W_lin, v_dis, v_heal)

        print(f"CausalDiffModelStub initialized:")
        print(f"  node_dim={node_dim}, edge_dim={edge_dim}, "
              f"cond_dim={cond_dim}, T={T}, scm_dim={scm_dim}")

    def diffuse(self, x: 'torch.Tensor', t: int) -> 'torch.Tensor':
        """Uniform categorical noise: lerp toward uniform at rate β_t."""
        n_cats = x.shape[-1]
        beta   = t / self.T
        noise  = torch.ones_like(x) / n_cats
        return (1 - beta) * x + beta * noise

    def compute_loss(self, node_feat: 'torch.Tensor',
                     edge_attr: 'torch.Tensor',
                     cond_vec: 'torch.Tensor',
                     lam: tuple = (0.4, 0.2, 0.3)) -> dict:
        """
        Full training loss for one batch.
        
        node_feat: [N, node_dim]    — ground-truth node features (one-hot)
        edge_attr: [E, edge_dim]    — ground-truth edge features
        cond_vec:  [batch, 128]     — pathway conditioning Δv
        lam:       (λ₁, λ₂, λ₃)    — loss weights
        
        Returns dict of individual losses and total.
        """
        # 1. Sample timestep
        t_val = torch.randint(1, self.T + 1, (1,)).item()

        # 2. Forward diffuse (add noise)
        x_t  = self.diffuse(node_feat, t_val)

        # 3. Encode with conditioning
        h    = self.encoder(x_t)
        # FiLM conditioning — broadcast cond_vec to all atoms
        # In real impl, use graph-level conditioning with proper batching
        cond_broadcast = cond_vec.mean(0, keepdim=True).expand(h.shape[0], -1)
        h    = self.cond_layer(h, cond_broadcast)

        # 4. Predict clean atom/bond types
        node_logits = self.node_head(h)
        edge_logits = self.edge_head(h.mean(0, keepdim=True).expand(edge_attr.shape[0], -1))

        # 5. Denoising loss (standard CE)
        node_target = node_feat.argmax(dim=-1)
        edge_target = edge_attr.argmax(dim=-1)
        L_denoise   = (F.cross_entropy(node_logits, node_target) +
                       F.cross_entropy(edge_logits, edge_target))

        # 6. Pathway consistency loss
        # Predicted graph-level representation vs conditioning vector
        graph_repr  = h.mean(0)  # global mean pool
        L_pathway   = F.mse_loss(
            nn.Linear(256, self.cond_dim)(graph_repr),
            cond_vec.mean(0)
        )

        # 7. ADMET penalty
        L_admet = self.admet_loss(node_logits)

        # 8. Causal consistency loss
        # delta_T: how this molecule shifts pathway nodes
        # In real impl: predict binding targets → compute pathway shift
        scm_dim  = self.causal_loss.M_inv.shape[0]
        delta_T  = torch.zeros(1, scm_dim)  # placeholder; replace with Chemprop
        L_causal = self.causal_loss(delta_T)

        # 9. Total weighted loss
        L_total = (L_denoise +
                   lam[0] * L_pathway +
                   lam[1] * L_admet +
                   lam[2] * L_causal)

        return {
            'L_total':   L_total.item(),
            'L_denoise': L_denoise.item(),
            'L_pathway': L_pathway.item(),
            'L_admet':   L_admet.item(),
            'L_causal':  L_causal.item(),
            't':         t_val,
        }


# ============================================================================
# SECTION 3: Setup Instructions
# ============================================================================

SETUP_INSTRUCTIONS = """
── Next steps to get Module C fully running ──────────────────────────────────

1. Clone DiGress (base diffusion model):
   git clone https://github.com/cvignac/DiGress
   cd DiGress && pip install -e . --break-system-packages

2. Install GPU dependencies:
   pip install torch==2.1.0+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install torch_geometric torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

3. Preprocess training data (using our 01_mol_pipeline.py):
   # Download ZINC15 subset (250k CNS-relevant, MW<450)
   wget https://zinc15.docking.org/tranches/head?subset=drugnow&output_fields=smiles -O zinc_cnssub.smi
   python scripts/01_mol_pipeline.py --input zinc_cnssub.smi --output data/zinc_graphs --bbb --limit 250000

   # Download ChEMBL30 (requires ~8GB disk)
   wget https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_30/chembl_30.sdf.gz

4. Get real Δv conditioning vectors:
   # After getting ADNI + CBTN data access (see 02_logbb_predictor.py comments):
   Rscript scripts/run_gsva.R  # outputs pathway_delta_v.csv

5. Integrate CausalConditioningLayer into DiGress:
   # In DiGress/src/diffusion/graph_transformer.py, after each attention block:
   self.cond_layers.append(CausalConditioningLayer(node_dim, cond_dim=128))

6. Train (requires A100 or similar):
   python train.py --config configs/causaldiff_dipg.yaml --gpus 4
   # Estimated: 150 epochs × 1.2M mols × 4 A100s ≈ 3-5 days
   # Cloud cost estimate: ~$400-800 on Lambda Labs / RunPod

Files produced by this step:
  models/causaldiff_dipg_checkpoint.pt   (3-5 days)
  models/causaldiff_ad_checkpoint.pt     (3-5 days, separately)
"""

print(SETUP_INSTRUCTIONS)


# ============================================================================
# SECTION 4: Architecture validation (CPU smoke test)
# ============================================================================

if __name__ == '__main__' and TORCH_AVAILABLE:
    print("── Running architecture validation (CPU) ──\n")

    torch.manual_seed(42)

    # Simulate a tiny molecule: 10 atoms, 18 bonds
    N, E = 10, 18
    node_feat = F.one_hot(torch.randint(0, 9, (N,)), 9).float()
    edge_attr = F.one_hot(torch.randint(0, 4, (E,)), 4).float()
    cond_vec  = torch.randn(1, 128)   # pathway Δv

    model = CausalDiffModelStub(scm_dim=64)
    losses = model.compute_loss(node_feat, edge_attr, cond_vec)

    print("\nLoss breakdown:")
    for k, v in losses.items():
        if k != 't':
            print(f"  {k:15s}: {v:.4f}")
    print(f"  timestep t    : {losses['t']}")

    print("\n✓ Architecture validation passed.")
    print(f"  Model params  : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  All losses finite: {all(np.isfinite(v) for k, v in losses.items() if k != 't')}")
    print("\nThis stub runs on CPU. Real training requires GPU + full DiGress + data.")
