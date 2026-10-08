"""Rank-k factorizations of a weight matrix under a calibration statistic.

standard   W_k = [W L]_k L^{-1}            with C = L L^T the (damped) input covariance
reweight   same, with C(lam) = lam * C_0 + (1 - lam) * C_rest for the key and value projections
deflate    W_k exact on the top eigenvector u of C_0, optimal among rank-k matrices with that property:
           M = W L,  q = L^{-1} u / ||L^{-1} u||,  M_k = M q q^T + [M (I - q q^T)]_{k-1},  W_k = M_k L^{-1}

All factorizations are returned as (A, B) with W_k = A @ B, A: (m, k), B: (k, n), computed in float64.
"""
from __future__ import annotations

from typing import Tuple

import torch

Tensor = torch.Tensor
RECIPES = ("standard", "reweight", "deflate")


def rank_for_removal(m: int, n: int, ratio: float) -> int:
    """Largest rank k whose factors A (m x k), B (k x n) store at most (1 - ratio) * m * n parameters."""
    return max(1, int((1 - ratio) * m * n / (m + n)))


def reweighted_statistic(C0: Tensor, C_rest: Tensor, lam: float) -> Tensor:
    """C(lam) = lam * C_0 + (1 - lam) * C_rest."""
    return lam * C0 + (1.0 - lam) * C_rest


def standard_statistic(C0: Tensor, C_rest: Tensor, seq_len: int) -> Tensor:
    """The ordinary input covariance: the mixture with lam = 1 / seq_len."""
    return reweighted_statistic(C0, C_rest, 1.0 / seq_len)


def sink_direction(X0: Tensor, directions: int = 1) -> Tensor:
    """Top eigenvector(s) of the first-position gram X0^T X0, as columns of an (n, directions) matrix."""
    _, evecs = torch.linalg.eigh(X0.to(torch.float64).T @ X0.to(torch.float64))
    return evecs[:, -directions:]


def damped_cholesky(C: Tensor, damp: float = 1e-3) -> Tensor:
    """Lower Cholesky factor of C + damp * mean(diag C) * I, escalating the damping until C is positive definite."""
    C = C.to(torch.float64).clone()
    jitter = damp * C.diagonal().mean()
    for attempt in range(8):
        C.diagonal().add_(jitter * (1 if attempt == 0 else 9 * 10 ** (attempt - 1)))
        L, info = torch.linalg.cholesky_ex(C)
        if info == 0:
            return L
    raise RuntimeError("covariance not positive definite even with heavy damping")


@torch.no_grad()
def whitened_factors(W: Tensor, C: Tensor, k: int, damp: float = 1e-3) -> Tuple[Tensor, Tensor]:
    """A @ B = [W L]_k L^{-1}: the rank-k matrix minimizing E||(W - W_k) x||^2 under C = E[x x^T]."""
    L = damped_cholesky(C.to(W.device), damp)
    U, S, Vh = torch.linalg.svd(W.detach().to(L) @ L, full_matrices=False)
    A = U[:, :k] * S[:k]
    B = torch.linalg.solve_triangular(L, Vh[:k], upper=False, left=False)  # B L = Vh  =>  B = Vh L^{-1}
    return A, B


@torch.no_grad()
def deflated_factors(W: Tensor, C: Tensor, X0: Tensor, k: int, damp: float = 1e-3, directions: int = 1) -> Tuple[Tensor, Tensor]:
    """Rank-k factors that reproduce W exactly on the top `directions` eigenvectors of X0^T X0 (the sink)."""
    directions = min(directions, k)
    L = damped_cholesky(C.to(W.device), damp)
    Wd = W.detach().to(L)
    U = sink_direction(X0.to(L), directions)
    Q, _ = torch.linalg.qr(torch.linalg.solve_triangular(L, U, upper=False))  # orthonormal basis of L^{-1} U
    M = Wd @ L
    MQ = M @ Q
    r = k - directions
    Us, S, Vh = torch.linalg.svd(M - MQ @ Q.T, full_matrices=False)
    A = torch.cat([MQ, Us[:, :r] * S[:r]], dim=1)          # (m, k)
    B_white = torch.cat([Q.T, Vh[:r]], dim=0)                # (k, n), acts on whitened inputs
    B = torch.linalg.solve_triangular(L, B_white, upper=False, left=False)  # B = B_white L^{-1}
    return A, B


@torch.no_grad()
def factors_for(W: Tensor, C0: Tensor, C_rest: Tensor, X0: Tensor, seq_len: int, k: int,
                recipe: str = "deflate", lam: float = 0.3, damp: float = 1e-3) -> Tuple[Tensor, Tensor]:
    """Dispatch on the recipe name. `standard` ignores lam and X0; `reweight` ignores X0."""
    if recipe == "standard":
        return whitened_factors(W, standard_statistic(C0, C_rest, seq_len), k, damp)
    if recipe == "reweight":
        return whitened_factors(W, reweighted_statistic(C0, C_rest, lam), k, damp)
    if recipe == "deflate":
        return deflated_factors(W, standard_statistic(C0, C_rest, seq_len), X0, k, damp)
    raise ValueError(f"unknown recipe {recipe!r}; choose from {RECIPES}")
