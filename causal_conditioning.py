"""
CausalDiffChem — causal_conditioning.py
========================================
Drop this file into your DiGress folder:
  DiGress/src/diffusion/causal_conditioning.py

This module provides:
  1. CausalConditioningLayer  — FiLM injection of pathway vector Δv
  2. PathwayEncoder           — GAT that projects Δv to 128-dim vector
  3. CausalConsistencyLoss    — differentiable L_causal via linearised SCM
  4. ADMETLoss                — soft BBB / MW / QED penalties
  5. CausalDiffLoss           — combines all four loss terms

Usage in DiGress training loop (see train_causal.py):
  from src.diffusion.causal_conditioning import (
      CausalConditioningLayer, CausalConsistencyLoss, CausalDiffLoss
  )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


# ── 1. FiLM Conditioning Layer ─────────────────────────────────────────────

class CausalConditioningLayer(nn.Module):
    """
    Injects pathway conditioning vector Δv into a graph transformer layer
    using Feature-wise Linear Modulation (FiLM).

    After each attention block in DiGress graph transformer, apply:
        h = self.cond_layer(h, delta_v)

    Args:
        node_dim:  dimension of node feature vectors h  (e.g. 256)
        cond_dim:  dimension of conditioning vector Δv  (default 128)

    Input:
        h:        [N_atoms, node_dim]  node features from attention block
        delta_v:  [batch_size, cond_dim]  pathway conditioning vector

    Output:
        h_cond:   [N_atoms, node_dim]  modulated node features
    """

    def __init__(self, node_dim: int, cond_dim: int = 128):
        super().__init__()
        # γ and β projections: cond_dim → node_dim
        self.gamma_proj = nn.Linear(cond_dim, node_dim)
        self.beta_proj  = nn.Linear(cond_dim, node_dim)
        self.norm       = nn.LayerNorm(node_dim)

        # initialise γ near 1, β near 0 (identity at start of training)
        nn.init.ones_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, h: torch.Tensor, delta_v: torch.Tensor) -> torch.Tensor:
        """
        h:        [N, node_dim]
        delta_v:  [B, cond_dim]  — one vector per graph in the batch

        In DiGress, h is batched via torch_geometric.
        We broadcast delta_v to match h using the batch index.
        For simplicity in the first integration, pass mean(delta_v) expanded.
        """
        # delta_v: [B, cond_dim] → use mean if batch > 1, else squeeze
        if delta_v.dim() == 2:
            cond = delta_v.mean(0, keepdim=True)  # [1, cond_dim]
        else:
            cond = delta_v.unsqueeze(0)            # [1, cond_dim]

        gamma = self.gamma_proj(cond)  # [1, node_dim]
        beta  = self.beta_proj(cond)   # [1, node_dim]

        # FiLM: scale and shift, then normalise
        return self.norm(gamma * h + beta)


# ── 2. Pathway Encoder (GAT) ───────────────────────────────────────────────

class PathwayEncoder(nn.Module):
    """
    Projects raw pathway conditioning vector Δv (variable dimension,
    one value per pathway gene set) into a fixed 128-dim vector.

    In full training: this is a 4-head GAT over the pathway interaction graph.
    For the initial integration: simple MLP projection (swap for GAT later).

    Args:
        input_dim:  number of pathway gene sets (128 DIPG, 144 AD)
        hidden_dim: intermediate MLP dimension
        output_dim: output conditioning dimension (must match CausalConditioningLayer cond_dim)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, delta_v_raw: torch.Tensor) -> torch.Tensor:
        """
        delta_v_raw: [B, input_dim]  raw pathway difference vector
        returns:     [B, 128]        encoded conditioning vector
        """
        return self.encoder(delta_v_raw)


# ── 3. Causal Consistency Loss ─────────────────────────────────────────────

class CausalConsistencyLoss(nn.Module):
    """
    L_causal: penalises generated molecules whose predicted binding profile
    fails to restore healthy network equilibrium.

    Uses the linearised SCM approximation:
        v_int = (I - W_lin)^{-1} · (v_disease + δ_T)
        L_causal = 1 - cosine_similarity(v_int, v_healthy)

    (I - W_lin)^{-1} is precomputed and frozen — gradients flow through δ_T.

    Args:
        W_lin:     [d, d]  NOTEARS adjacency matrix (numpy array or tensor)
        v_disease: [d]     mean disease pathway state vector
        v_healthy: [d]     mean healthy reference pathway state vector

    Call:
        loss = causal_loss(delta_T)
        where delta_T: [B, d] — predicted binding-induced pathway perturbation
    """

    def __init__(
        self,
        W_lin:     np.ndarray,
        v_disease: np.ndarray,
        v_healthy: np.ndarray,
    ):
        super().__init__()
        d = W_lin.shape[0]
        I = np.eye(d)

        # Precompute M_inv = (I - W_lin)^{-1}
        try:
            M_inv = np.linalg.solve(I - W_lin, I)
        except np.linalg.LinAlgError:
            # Fallback: use pseudoinverse if W is near-singular
            M_inv = np.linalg.pinv(I - W_lin)

        # Register as buffers (moved to GPU automatically with .to(device))
        self.register_buffer('M_inv',     torch.tensor(M_inv,     dtype=torch.float32))
        self.register_buffer('v_disease', torch.tensor(v_disease, dtype=torch.float32))
        self.register_buffer('v_healthy', torch.tensor(v_healthy, dtype=torch.float32))

    def forward(self, delta_T: torch.Tensor) -> torch.Tensor:
        """
        delta_T: [B, d]  predicted intervention effect
                         (how each generated molecule perturbs SCM nodes)

        Returns: scalar loss
        """
        # Post-intervention state
        # v_int = (v_disease + delta_T) @ M_inv.T   [B, d]
        v_base = self.v_disease.unsqueeze(0) + delta_T        # [B, d]
        v_int  = v_base @ self.M_inv.T                         # [B, d]

        # Phenotype reversal score R = cos_sim(v_int, v_healthy)
        R = F.cosine_similarity(
            v_int,
            self.v_healthy.unsqueeze(0).expand_as(v_int),
            dim=1
        )  # [B]

        # Loss: maximise R → minimise 1 - R
        return (1.0 - R).mean()

    def reversal_score(self, delta_T: torch.Tensor) -> torch.Tensor:
        """Compute R without loss sign — useful for evaluation."""
        v_base = self.v_disease.unsqueeze(0) + delta_T
        v_int  = v_base @ self.M_inv.T
        return F.cosine_similarity(
            v_int,
            self.v_healthy.unsqueeze(0).expand_as(v_int),
            dim=1
        )


# ── 4. ADMET Loss ──────────────────────────────────────────────────────────

class ADMETLoss(nn.Module):
    """
    Soft penalties for violating CNS drug-likeness constraints.
    Applied to generated molecule property predictions.

    Penalties are soft ReLU (differentiable), not hard constraints.

    Properties checked:
        logBB < threshold  → BBB impermeability penalty
        MW    > mw_limit   → size penalty
        QED   < qed_thresh → drug-likeness penalty

    In full training: use differentiable property predictors.
    In stub mode: returns 0 (replace with real predictors).
    """

    def __init__(
        self,
        logbb_threshold: float = -1.0,
        mw_limit:        float = 500.0,
        qed_threshold:   float = 0.5,
        weights: tuple = (1.0, 0.5, 1.0),
    ):
        super().__init__()
        self.logbb_threshold = logbb_threshold
        self.mw_limit        = mw_limit
        self.qed_threshold   = qed_threshold
        self.w_bbb, self.w_mw, self.w_qed = weights

    def forward(
        self,
        logbb_pred: Optional[torch.Tensor] = None,
        mw_pred:    Optional[torch.Tensor] = None,
        qed_pred:   Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Each input: [B] tensor of predicted property values, or None.
        Returns scalar penalty.
        """
        loss = torch.tensor(0.0, requires_grad=True)

        if logbb_pred is not None:
            bbb_pen = F.relu(self.logbb_threshold - logbb_pred).mean()
            loss = loss + self.w_bbb * bbb_pen

        if mw_pred is not None:
            mw_pen = F.relu(mw_pred - self.mw_limit).mean() / self.mw_limit
            loss = loss + self.w_mw * mw_pen

        if qed_pred is not None:
            qed_pen = F.relu(self.qed_threshold - qed_pred).mean()
            loss = loss + self.w_qed * qed_pen

        return loss


# ── 5. Combined CausalDiff Loss ────────────────────────────────────────────

class CausalDiffLoss(nn.Module):
    """
    Full CausalDiffChem training objective:

        L_total = L_denoise + λ₁·L_pathway + λ₂·L_ADMET + λ₃·L_causal

    Args:
        causal_loss:  CausalConsistencyLoss instance (loaded with SCM data)
        admet_loss:   ADMETLoss instance
        lambda1:      weight for L_pathway  (default 0.40)
        lambda2:      weight for L_ADMET    (default 0.20)
        lambda3:      weight for L_causal   (default 0.30)
    """

    def __init__(
        self,
        causal_loss: CausalConsistencyLoss,
        admet_loss:  ADMETLoss,
        lambda1: float = 0.40,
        lambda2: float = 0.20,
        lambda3: float = 0.30,
    ):
        super().__init__()
        self.causal_loss = causal_loss
        self.admet_loss  = admet_loss
        self.lambda1     = lambda1
        self.lambda2     = lambda2
        self.lambda3     = lambda3

    def forward(
        self,
        l_denoise:    torch.Tensor,          # from DiGress base model
        graph_repr:   torch.Tensor,          # [B, node_dim] global pooled
        delta_v_hat:  torch.Tensor,          # [B, 128] target conditioning vec
        delta_T:      torch.Tensor,          # [B, d] binding-induced perturbation
        logbb_pred:   Optional[torch.Tensor] = None,
        mw_pred:      Optional[torch.Tensor] = None,
        qed_pred:     Optional[torch.Tensor] = None,
        cond_proj:    Optional[nn.Linear]    = None,  # projects graph_repr → 128
    ) -> dict:
        """
        Returns dict with individual losses and total.
        """
        # L_pathway: generated molecule representation vs target Δv
        if cond_proj is not None:
            mol_repr = cond_proj(graph_repr)   # [B, 128]
        else:
            mol_repr = graph_repr[:, :delta_v_hat.shape[-1]]
        l_pathway = F.mse_loss(mol_repr, delta_v_hat)

        # L_ADMET
        l_admet = self.admet_loss(logbb_pred, mw_pred, qed_pred)

        # L_causal
        l_causal = self.causal_loss(delta_T)

        # Total
        l_total = (
            l_denoise
            + self.lambda1 * l_pathway
            + self.lambda2 * l_admet
            + self.lambda3 * l_causal
        )

        return {
            'total':    l_total,
            'denoise':  l_denoise.detach(),
            'pathway':  l_pathway.detach(),
            'admet':    l_admet.detach(),
            'causal':   l_causal.detach(),
        }
