"""
CausalDiffChem — causal_dataset.py
====================================
Drop this file into your DiGress folder:
  DiGress/src/datasets/causal_dataset.py

PyTorch Dataset that:
  1. Loads pre-processed molecular graph .npz files (from 01_mol_pipeline.py)
  2. Loads the pathway conditioning vector Δv (from delta_v_dipg.npy or delta_v_ad.npy)
  3. Returns (graph, delta_v) pairs for training

Usage:
  dataset = CausalMolDataset(
      graph_dir='graphs/',
      delta_v_path='delta_v_dipg.npy',
      disease='dipg'
  )
  loader = DataLoader(dataset, batch_size=128, shuffle=True)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class CausalMolDataset(Dataset):
    """
    Dataset of molecular graphs paired with a pathway conditioning vector.

    Each item returns a dict:
        x:          [N_atoms, 9]   node feature tensor
        edge_index: [2, N_edges]   bond connectivity
        edge_attr:  [N_edges, 4]   bond type one-hot
        delta_v:    [128]          pathway conditioning vector (same for all)
        smiles:     str            canonical SMILES
        qed:        float          drug-likeness score
        mw:         float          molecular weight

    Args:
        graph_dir:    path to directory containing mol_*.npz files
        delta_v_path: path to .npy file containing Δv vector
        max_atoms:    skip molecules with more than this many atoms (default 50)
        limit:        optionally limit dataset size (useful for testing)
    """

    def __init__(
        self,
        graph_dir:    str,
        delta_v_path: str,
        max_atoms:    int = 50,
        limit:        int = None,
    ):
        self.graph_dir = Path(graph_dir)
        self.max_atoms = max_atoms

        # Load conditioning vector — same for every molecule in this dataset
        delta_v_raw = np.load(delta_v_path)
        self.delta_v = torch.tensor(delta_v_raw, dtype=torch.float32)

        # If delta_v is longer than 128, truncate; if shorter, pad with zeros
        target_dim = 128
        if self.delta_v.shape[0] > target_dim:
            self.delta_v = self.delta_v[:target_dim]
        elif self.delta_v.shape[0] < target_dim:
            pad = torch.zeros(target_dim - self.delta_v.shape[0])
            self.delta_v = torch.cat([self.delta_v, pad])

        # Normalise Δv to unit norm
        norm = self.delta_v.norm()
        if norm > 0:
            self.delta_v = self.delta_v / norm

        # Collect all .npz file paths
        all_files = sorted(self.graph_dir.glob('mol_*.npz'))
        if limit:
            all_files = all_files[:limit]

        # Filter by max_atoms
        self.files = []
        for f in all_files:
            try:
                d = np.load(f, allow_pickle=True)
                n_atoms = d['x'].shape[0]
                if n_atoms <= max_atoms:
                    self.files.append(f)
            except Exception:
                continue

        print(f"CausalMolDataset: {len(self.files):,} molecules loaded")
        print(f"  Δv shape: {self.delta_v.shape}  norm: {self.delta_v.norm():.4f}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = np.load(self.files[idx], allow_pickle=True)

        x          = torch.tensor(d['x'],          dtype=torch.float32)
        edge_index = torch.tensor(d['edge_index'],  dtype=torch.long)
        edge_attr  = torch.tensor(d['edge_attr'],   dtype=torch.float32)
        qed        = float(d['qed'].item())
        mw         = float(d['mw'].item())
        smiles     = str(d['smiles'][0])

        return {
            'x':          x,
            'edge_index': edge_index,
            'edge_attr':  edge_attr,
            'delta_v':    self.delta_v,   # same for all samples in this disease
            'smiles':     smiles,
            'qed':        torch.tensor(qed, dtype=torch.float32),
            'mw':         torch.tensor(mw,  dtype=torch.float32),
            'n_atoms':    x.shape[0],
        }


def collate_graphs(batch):
    """
    Custom collate function for variable-size molecular graphs.
    Stacks graphs into a batched format compatible with PyTorch Geometric.
    """
    # Stack conditioning vectors — all same, just pick first
    delta_v   = batch[0]['delta_v'].unsqueeze(0).expand(len(batch), -1).clone()  # [B, 128]
    qed       = torch.stack([b['qed'] for b in batch])
    mw        = torch.stack([b['mw']  for b in batch])
    n_atoms   = [b['n_atoms'] for b in batch]
    smiles    = [b['smiles']  for b in batch]

    # Build batched node/edge tensors with batch_index for pooling
    x_list, ei_list, ea_list = [], [], []
    offset = 0
    batch_index = []

    for i, b in enumerate(batch):
        x  = b['x']
        ei = b['edge_index'] + offset
        ea = b['edge_attr']
        n  = x.shape[0]

        x_list.append(x)
        ei_list.append(ei)
        ea_list.append(ea)
        batch_index.extend([i] * n)
        offset += n

    return {
        'x':           torch.cat(x_list,  dim=0),     # [total_atoms, 9]
        'edge_index':  torch.cat(ei_list, dim=1),     # [2, total_edges]
        'edge_attr':   torch.cat(ea_list, dim=0),     # [total_edges, 4]
        'batch':       torch.tensor(batch_index, dtype=torch.long),
        'delta_v':     delta_v,                        # [B, 128]
        'qed':         qed,                            # [B]
        'mw':          mw,                             # [B]
        'n_atoms':     n_atoms,
        'smiles':      smiles,
    }


def build_dataloaders(
    graph_dir:    str,
    delta_v_path: str,
    batch_size:   int = 128,
    val_split:    float = 0.05,
    num_workers:  int = 4,
    limit:        int = None,
    max_atoms:    int = 50,
):
    """
    Build train and validation DataLoaders.

    Returns:
        train_loader, val_loader
    """
    dataset = CausalMolDataset(
        graph_dir=graph_dir,
        delta_v_path=delta_v_path,
        max_atoms=max_atoms,
        limit=limit,
    )

    n_val   = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val

    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_graphs,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Train: {n_train:,} molecules  |  Val: {n_val:,} molecules")
    print(f"Train batches: {len(train_loader):,}  |  Val batches: {len(val_loader):,}")

    return train_loader, val_loader


if __name__ == '__main__':
    # Quick test — run from causaldiffchem/ folder
    import sys
    graph_dir    = sys.argv[1] if len(sys.argv) > 1 else 'graphs/'
    delta_v_path = sys.argv[2] if len(sys.argv) > 2 else 'delta_v_dipg.npy'

    train_loader, val_loader = build_dataloaders(
        graph_dir=graph_dir,
        delta_v_path=delta_v_path,
        batch_size=16,
        limit=100,
        num_workers=0,
    )

    batch = next(iter(train_loader))
    print("\nSample batch:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:12s}: {v.shape}  dtype={v.dtype}")
        elif isinstance(v, list):
            print(f"  {k:12s}: list of {len(v)}")
    print("\nDataset loader working correctly.")
