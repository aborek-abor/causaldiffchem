"""
notears_gpu.py  —  GPU-accelerated NOTEARS for CausalDiffChem
==============================================================
Replaces scipy L-BFGS-B + numerical finite-difference gradients with
PyTorch autograd + CUDA.  Analytical gradients are ~400x faster for a
20-gene (400-parameter) problem because scipy was calling the objective
function 400 times per gradient step via approx_derivative().

USAGE
-----
In proper_notears.py, add ONE line at the very top of the file:

    from notears_gpu import notears_gpu as notears_proper   # GPU override

Python will use this definition instead of the local notears_proper().
If CUDA is unavailable the function falls back to CPU automatically.

CLI stays identical:
    py -3.11 proper_notears.py --disease tuberculosis_clean
"""

import numpy as np
import torch


# ──────────────────────────────────────────────────────────────────────────────
def notears_gpu(
    X,
    lambda1: float = 0.05,
    max_iter: int   = 300,
    h_tol: float    = 1e-8,
    rho_max: float  = 1e16,
    lr: float       = 1.0,
) -> np.ndarray:
    """
    GPU-accelerated NOTEARS with PyTorch autograd.

    Parameters
    ----------
    X        : (n_samples, n_genes) float64 numpy array, z-scored expression
    lambda1  : L1 sparsity coefficient          (default 0.05)
    max_iter : outer augmented-Lagrangian iters  (default 300)
    h_tol    : acyclicity tolerance              (default 1e-8)
    rho_max  : max penalty coefficient           (default 1e16)
    lr       : LBFGS learning rate               (default 1.0)

    Returns
    -------
    W : (n_genes, n_genes) float64 numpy array — DAG weight matrix,
        edges with |w| < 0.05 zeroed (matches existing pipeline threshold)
    """
    # ── Device selection ──────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float64          # float64 for numerical stability of expm

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  [GPU] {gpu_name}  ({vram_gb:.1f} GB VRAM)  — analytical gradients active")
    else:
        print("  [GPU] CUDA not available — running on CPU with autograd")

    n, d = X.shape
    X_t  = torch.tensor(X, dtype=dtype, device=device)

    # ── Core functions ────────────────────────────────────────────────────────
    def h_func(W_):
        """Acyclicity constraint: tr(e^{W ⊙ W}) - d"""
        return torch.trace(torch.linalg.matrix_exp(W_ * W_)) - d

    def augmented_lagrangian(W_, rho_, alpha_):
        """MSE + L1 + alpha*h + (rho/2)*h² """
        resid = X_t - X_t @ W_
        mse   = 0.5 / n * (resid * resid).sum()
        l1    = lambda1 * W_.abs().sum()
        h     = h_func(W_)
        aug   = alpha_ * h + 0.5 * rho_ * h * h
        return mse + l1 + aug, h

    # ── Initialise ────────────────────────────────────────────────────────────
    W      = torch.zeros(d, d, dtype=dtype, device=device)
    rho    = 1.0
    alpha  = 0.0
    h_prev = np.inf

    # ── Outer augmented-Lagrangian loop ───────────────────────────────────────
    for outer in range(max_iter):

        W_var = W.detach().clone().requires_grad_(True)

        opt = torch.optim.LBFGS(
            [W_var],
            lr                = lr,
            max_iter          = 100,
            tolerance_grad    = 1e-7,
            tolerance_change  = 1e-9,
            line_search_fn    = "strong_wolfe",
        )

        def closure():
            opt.zero_grad()
            loss, _ = augmented_lagrangian(W_var, rho, alpha)
            loss.backward()
            with torch.no_grad():
                W_var.grad.fill_diagonal_(0.0)   # no self-loops in gradient
            return loss

        opt.step(closure)

        # Enforce no self-loops in W
        with torch.no_grad():
            W_var.fill_diagonal_(0.0)
            h_val   = h_func(W_var).item()
            n_edges = int((W_var.abs() > 1e-4).sum().item())
            mse_val = float(0.5 / n * ((X_t - X_t @ W_var) ** 2).sum().item())

        print(f"  iter {outer:3d}: h={h_val:.6f}  edges={n_edges}  loss={mse_val:.4f}")

        # ── Convergence ───────────────────────────────────────────────────────
        if h_val <= h_tol:
            print(f"  [GPU] Converged  h={h_val:.2e} ≤ tol={h_tol:.2e}")
            break

        # ── Dual / penalty update ─────────────────────────────────────────────
        alpha  = alpha + rho * h_val
        if h_val > 0.25 * h_prev:
            rho = min(rho * 10.0, rho_max)
        h_prev = h_val
        W      = W_var.detach()

    # ── Threshold and convert to numpy ────────────────────────────────────────
    W_np = W_var.detach().cpu().numpy()
    W_np[np.abs(W_np) < 0.05] = 0.0        # match existing pipeline threshold
    np.fill_diagonal(W_np, 0.0)

    n_final = int(np.sum(np.abs(W_np) > 0))
    print(f"  [GPU] Done — {n_final} edges retained (|w| > 0.05)")
    return W_np


# ── Alias so the import override is transparent ───────────────────────────────
notears_proper = notears_gpu


# ── Quick sanity test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    n_s, n_g = 92, 20
    X_test = np.random.randn(n_s, n_g)
    print(f"Sanity test: {n_s} samples, {n_g} genes")
    W_test = notears_gpu(X_test, lambda1=0.05)
    print(f"W shape: {W_test.shape}  |  edges: {int(np.sum(np.abs(W_test) > 0))}")
    print("PASS")
