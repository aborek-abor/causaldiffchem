"""
CausalDiffChem — train_causal.py
==================================
Place this file in your causaldiffchem/ folder (same level as DiGress/).

This is the main training script. Run it like this:

  DIPG training:
    py -3.11 train_causal.py --disease dipg --epochs 150 --batch_size 128

  AD training:
    py -3.11 train_causal.py --disease ad --epochs 150 --batch_size 128

  Quick test (CPU, small batch):
    py -3.11 train_causal.py --disease dipg --epochs 2 --batch_size 8 --limit 500 --no_gpu

What this script does:
  1. Loads 152,559 molecular graphs from graphs/
  2. Loads disease conditioning vector (delta_v_dipg.npy or delta_v_ad.npy)
  3. Loads disease SCM (W_dipg.npy or W_ad.npy)
  4. Builds a conditioned diffusion model (DiGress + FiLM layers)
  5. Trains with L_total = L_denoise + λ1·L_pathway + λ2·L_ADMET + λ3·L_causal
  6. Saves checkpoints every 10 epochs to models/
  7. Logs losses to training_log.csv

Requirements:
  pip install torch torch_geometric pytorch_lightning tqdm numpy
  DiGress cloned to DiGress/ subfolder

GPU: your RTX 3000 will work for small/medium runs.
     Full 150-epoch run on 152k molecules: ~8-12 hours on RTX 3000.
"""

import os
import sys
import csv
import time
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add DiGress to path
sys.path.insert(0, str(Path(__file__).parent / 'DiGress' / 'src'))

from causal_conditioning import (
    CausalConditioningLayer,
    PathwayEncoder,
    CausalConsistencyLoss,
    ADMETLoss,
    CausalDiffLoss,
)
from causal_dataset import build_dataloaders


# ── Argument parser ────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='CausalDiffChem Training')
    p.add_argument('--disease', type=str, default='dipg', choices=['dipg', 'ad', 'ms_blood', 'hiv', 'tuberculosis', 'leishmaniasis', 'leishmaniasis2', 'malaria2', 'sarcoidosis2', 'dengue', 'breast_cancer', 'breast_cancer2'])
    p.add_argument('--graph_dir',  type=str, default='graphs/')
    p.add_argument('--data_dir',   type=str, default='data/',
                   help='Directory containing delta_v_*.npy and W_*.npy files')
    p.add_argument('--model_dir',  type=str, default='models/')
    p.add_argument('--epochs',     type=int, default=150)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--node_dim',   type=int, default=256)
    p.add_argument('--cond_dim',   type=int, default=128)
    p.add_argument('--T',          type=int, default=1000,
                   help='Diffusion timesteps')
    p.add_argument('--lambda1',    type=float, default=0.40)
    p.add_argument('--lambda2',    type=float, default=0.20)
    p.add_argument('--lambda3',    type=float, default=0.30)
    p.add_argument('--limit',      type=int, default=None,
                   help='Limit dataset size (for testing)')
    p.add_argument('--no_gpu',     action='store_true')
    p.add_argument('--num_workers',type=int, default=4)
    p.add_argument('--save_every', type=int, default=10,
                   help='Save checkpoint every N epochs')
    p.add_argument('--resume',     type=str, default=None,
                   help='Path to checkpoint to resume from')
    return p.parse_args()


# ── Minimal graph transformer stub ────────────────────────────────────────
# Replace this with the full DiGress GraphTransformer once integrated

class MinimalGraphTransformer(nn.Module):
    """
    Placeholder graph transformer for architecture validation.
    Replace with DiGress's actual GraphTransformer in full training.

    In full integration:
      from diffusion.graph_transformer import GraphTransformer
      self.transformer = GraphTransformer(...)
      # Add CausalConditioningLayer after each attention block
    """

    def __init__(self, node_dim: int, edge_dim: int, cond_dim: int, n_layers: int = 4):
        super().__init__()
        self.node_dim = node_dim
        self.cond_dim = cond_dim

        # Node embedding
        self.node_embed = nn.Linear(9, node_dim)

        # Simplified transformer layers (replace with real attention)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(node_dim, node_dim),
                nn.ReLU(),
                nn.Linear(node_dim, node_dim),
            )
            for _ in range(n_layers)
        ])

        # FiLM conditioning after each layer
        self.cond_layers = nn.ModuleList([
            CausalConditioningLayer(node_dim, cond_dim)
            for _ in range(n_layers)
        ])

        # Output heads
        self.node_head = nn.Linear(node_dim, 9)    # predict clean atom types
        self.edge_head = nn.Linear(node_dim, 4)    # predict clean bond types

        # Projection for pathway consistency loss
        self.pathway_proj = nn.Linear(node_dim, cond_dim)

    def forward(self, x, edge_index, edge_attr, delta_v, t_embed=None):
        """
        x:          [N, 9]       noised node features
        edge_index: [2, E]       bond connectivity
        edge_attr:  [E, 4]       noised edge features
        delta_v:    [B, 128]     pathway conditioning vector
        t_embed:    [B, node_dim] timestep embedding (optional)

        Returns:
            node_logits: [N, 9]    predicted clean atom types
            edge_logits: [E, 4]    predicted clean bond types
            graph_repr:  [B, node_dim] global graph representation
        """
        h = self.node_embed(x)  # [N, node_dim]

        for layer, cond_layer in zip(self.layers, self.cond_layers):
            h = layer(h)
            h = cond_layer(h, delta_v)

        node_logits = self.node_head(h)   # [N, 9]

        # Simple edge logits from mean of endpoint features
        src, dst = edge_index
        edge_feat = (h[src] + h[dst]) / 2
        edge_logits = self.edge_head(edge_feat)   # [E, 4]

        # Global graph representation via mean pooling
        # In full PyG integration: use global_mean_pool(h, batch)
        graph_repr = h.mean(0, keepdim=True).expand(delta_v.shape[0], -1)  # [B, node_dim]

        return node_logits, edge_logits, graph_repr


# ── Diffusion utilities ────────────────────────────────────────────────────

def uniform_noise(x: torch.Tensor, t: int, T: int) -> torch.Tensor:
    """
    Forward diffusion: interpolate between clean x and uniform distribution.
    x: [N, C] one-hot or soft
    """
    n_cats = x.shape[-1]
    beta   = t / T
    noise  = torch.ones_like(x) / n_cats
    return (1 - beta) * x + beta * noise


def sample_timestep(batch_size: int, T: int, device) -> torch.Tensor:
    return torch.randint(1, T + 1, (batch_size,), device=device)


# ── Training loop ──────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, loss_fn, device, T, args):
    model.train()
    totals = {'total': 0, 'denoise': 0, 'pathway': 0, 'admet': 0, 'causal': 0}
    n_batches = 0

    pbar = tqdm(loader, desc='  train', leave=False)
    for batch in pbar:
        x          = batch['x'].to(device)
        edge_index = batch['edge_index'].to(device)
        edge_attr  = batch['edge_attr'].to(device)
        delta_v    = batch['delta_v'].to(device)
        qed        = batch['qed'].to(device)
        mw         = batch['mw'].to(device)

        # Sample timestep
        t_val = torch.randint(1, T + 1, (1,)).item()

        # Forward diffuse
        x_t   = uniform_noise(x,         t_val, T)
        ea_t  = uniform_noise(edge_attr,  t_val, T)

        # Forward pass
        node_logits, edge_logits, graph_repr = model(
            x_t, edge_index, ea_t, delta_v
        )

        # L_denoise
        node_target = x.argmax(dim=-1)
        edge_target = edge_attr.argmax(dim=-1)
        l_denoise = (
            F.cross_entropy(node_logits, node_target) +
            F.cross_entropy(edge_logits, edge_target)
        )

        # delta_T placeholder — zeros until Chemprop is integrated
        scm_dim = loss_fn.causal_loss.M_inv.shape[0]
        delta_T = torch.zeros(delta_v.shape[0], scm_dim, device=device)

        # Full loss
        losses = loss_fn(
            l_denoise   = l_denoise,
            graph_repr  = graph_repr,
            delta_v_hat = delta_v,
            delta_T     = delta_T,
            qed_pred    = qed,
            mw_pred     = mw,
            cond_proj   = model.pathway_proj,
        )

        optimizer.zero_grad()
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        for k in totals:
            totals[k] += losses[k].item()
        n_batches += 1

        pbar.set_postfix({
            'L': f"{losses['total'].item():.3f}",
            'dn': f"{losses['denoise'].item():.3f}",
            'ca': f"{losses['causal'].item():.3f}",
        })

    return {k: v / n_batches for k, v in totals.items()}


@torch.no_grad()
def val_epoch(model, loader, loss_fn, device, T):
    model.eval()
    totals = {'total': 0, 'denoise': 0, 'pathway': 0, 'admet': 0, 'causal': 0}
    n_batches = 0

    for batch in loader:
        x          = batch['x'].to(device)
        edge_index = batch['edge_index'].to(device)
        edge_attr  = batch['edge_attr'].to(device)
        delta_v    = batch['delta_v'].to(device)
        qed        = batch['qed'].to(device)
        mw         = batch['mw'].to(device)

        t_val = torch.randint(1, T + 1, (1,)).item()
        x_t   = uniform_noise(x, t_val, T)
        ea_t  = uniform_noise(edge_attr, t_val, T)

        node_logits, edge_logits, graph_repr = model(x_t, edge_index, ea_t, delta_v)

        node_target = x.argmax(dim=-1)
        edge_target = edge_attr.argmax(dim=-1)
        l_denoise = (
            F.cross_entropy(node_logits, node_target) +
            F.cross_entropy(edge_logits, edge_target)
        )

        scm_dim = loss_fn.causal_loss.M_inv.shape[0]
        delta_T = torch.zeros(delta_v.shape[0], scm_dim, device=device)

        losses = loss_fn(
            l_denoise   = l_denoise,
            graph_repr  = graph_repr,
            delta_v_hat = delta_v,
            delta_T     = delta_T,
            qed_pred    = qed,
            mw_pred     = mw,
            cond_proj   = model.pathway_proj,
        )

        for k in totals:
            totals[k] += losses[k].item()
        n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Device
    if args.no_gpu or not torch.cuda.is_available():
        device = torch.device('cpu')
        print("Running on CPU")
    else:
        device = torch.device('cuda')
        print(f"Running on GPU: {torch.cuda.get_device_name(0)}")

    # Paths
    data_dir     = Path(args.data_dir)
    model_dir    = Path(args.model_dir)
    model_dir.mkdir(exist_ok=True)

    delta_v_path = data_dir / f'delta_v_{args.disease}.npy'
    W_path       = data_dir / f'W_{args.disease}.npy'
    v_dis_path   = data_dir / f'v_disease_{args.disease}.npy'
    v_heal_path  = data_dir / f'v_healthy_{args.disease}.npy'

    # Validate paths
    for p in [delta_v_path, W_path, v_dis_path, v_heal_path]:
        if not p.exists():
            print(f"WARNING: {p} not found. Run the SCM construction script first.")
            print("Using random placeholders for testing.")

    # Load SCM data
    scm_dim = 20   # default for testing
    if W_path.exists():
        W        = np.load(W_path)
        v_dis    = np.load(v_dis_path)
        v_heal   = np.load(v_heal_path)
        scm_dim  = W.shape[0]
        v_dis    = v_dis[:scm_dim]
        v_heal   = v_heal[:scm_dim]
        print(f"SCM loaded: {scm_dim} nodes, {int((np.abs(W)>0.05).sum())} edges")
    else:
        W        = np.zeros((scm_dim, scm_dim))
        v_dis    = np.zeros(scm_dim)
        v_heal   = np.ones(scm_dim) * 0.1

    # Build dataloaders
    print(f"\nLoading {args.disease.upper()} dataset...")
    if delta_v_path.exists():
        dv_path = str(delta_v_path)
    else:
        # Create placeholder if missing
        dv = np.random.randn(args.cond_dim).astype(np.float32)
        dv_path = str(model_dir / 'delta_v_placeholder.npy')
        np.save(dv_path, dv)
        print(f"  Using placeholder Δv (replace with real data)")

    train_loader, val_loader = build_dataloaders(
        graph_dir    = args.graph_dir,
        delta_v_path = dv_path,
        batch_size   = args.batch_size,
        limit        = args.limit,
        num_workers  = args.num_workers if not args.no_gpu else 0,
    )

    # Build model
    print(f"\nBuilding CausalDiffModel...")
    model = MinimalGraphTransformer(
        node_dim = args.node_dim,
        edge_dim = 4,
        cond_dim = args.cond_dim,
        n_layers = 4,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Build loss
    causal_loss = CausalConsistencyLoss(W, v_dis, v_heal).to(device)
    admet_loss  = ADMETLoss().to(device)
    loss_fn     = CausalDiffLoss(
        causal_loss = causal_loss,
        admet_loss  = admet_loss,
        lambda1     = args.lambda1,
        lambda2     = args.lambda2,
        lambda3     = args.lambda3,
    )

    # Optimiser
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Resume from checkpoint
    start_epoch = 1
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from epoch {ckpt['epoch']}")

    # Log file
    log_path = model_dir / f'training_log_{args.disease}.csv'
    log_fields = ['epoch', 'train_total', 'train_denoise', 'train_pathway',
                  'train_admet', 'train_causal', 'val_total', 'time_s']

    with open(log_path, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    # ── Training loop ──────────────────────────────────────────────────────
    print(f"\nTraining CausalDiffChem ({args.disease.upper()}) for {args.epochs} epochs")
    print(f"  λ1={args.lambda1}  λ2={args.lambda2}  λ3={args.lambda3}")
    print(f"  T={args.T}  batch={args.batch_size}  lr={args.lr}")
    print("-" * 60)

    best_val = float('inf')

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_losses = train_epoch(model, train_loader, optimizer, loss_fn,
                                   device, args.T, args)
        val_losses   = val_epoch(model, val_loader, loss_fn, device, args.T)
        scheduler.step()

        elapsed = time.time() - t0

        # Print
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train={train_losses['total']:.4f}  "
            f"(dn={train_losses['denoise']:.3f} "
            f"pw={train_losses['pathway']:.3f} "
            f"ca={train_losses['causal']:.3f})  "
            f"val={val_losses['total']:.4f}  "
            f"[{elapsed:.0f}s]"
        )

        # Log
        with open(log_path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=log_fields).writerow({
                'epoch':          epoch,
                'train_total':    f"{train_losses['total']:.6f}",
                'train_denoise':  f"{train_losses['denoise']:.6f}",
                'train_pathway':  f"{train_losses['pathway']:.6f}",
                'train_admet':    f"{train_losses['admet']:.6f}",
                'train_causal':   f"{train_losses['causal']:.6f}",
                'val_total':      f"{val_losses['total']:.6f}",
                'time_s':         f"{elapsed:.1f}",
            })

        # Save checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = model_dir / f'causaldiff_{args.disease}_epoch{epoch:03d}.pt'
            torch.save({
                'epoch':     epoch,
                'model':     model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args':      vars(args),
                'val_loss':  val_losses['total'],
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")

        # Save best
        if val_losses['total'] < best_val:
            best_val = val_losses['total']
            best_path = model_dir / f'causaldiff_{args.disease}_best.pt'
            torch.save({
                'epoch':    epoch,
                'model':    model.state_dict(),
                'val_loss': best_val,
            }, best_path)

    print(f"\nTraining complete. Best val loss: {best_val:.4f}")
    print(f"Best model: {model_dir}/causaldiff_{args.disease}_best.pt")
    print(f"Training log: {log_path}")


if __name__ == '__main__':
    main()
