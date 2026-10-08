"""Synthetic tests of the factorizations: no model, no download, CPU only."""
import torch

from sinkdeflation import (deflated_factors, rank_for_removal, reweighted_statistic, sink_direction, standard_statistic,
                           whitened_factors)

torch.manual_seed(0)
M, N, T, WINDOWS = 96, 64, 128, 40


def _synthetic():
    """Inputs whose position 0 is a nearly fixed 'sink' direction and whose other positions are generic."""
    sink = torch.randn(N)
    sink = sink / sink.norm()
    x = torch.randn(WINDOWS, T, N)
    x[:, 0] = 6.0 * sink + 0.05 * torch.randn(WINDOWS, N)
    X0 = x[:, 0]
    rest = x[:, 1:].reshape(-1, N)
    return x, X0, X0.T @ X0 / WINDOWS, rest.T @ rest / rest.shape[0]


def _errors(W, A, B, X0, rest):
    Wk = (A @ B).to(W.dtype)
    e0 = ((W - Wk) @ X0.T).norm() / (W @ X0.T).norm()
    er = ((W - Wk) @ rest.T).norm() / (W @ rest.T).norm()
    return e0.item(), er.item()


def test_standard_statistic_is_the_plain_covariance():
    x, X0, C0, C_rest = _synthetic()
    plain = x.reshape(-1, N).T @ x.reshape(-1, N) / (WINDOWS * T)
    assert torch.allclose(standard_statistic(C0, C_rest, T), plain, atol=1e-4)


def test_rank_rule_respects_the_parameter_budget():
    for m, n, r in [(96, 64, 0.2), (4096, 4096, 0.4), (11008, 4096, 0.2)]:
        k = rank_for_removal(m, n, r)
        assert k * (m + n) <= (1 - r) * m * n < (k + 1) * (m + n)


def test_whitened_factors_shapes_and_optimality():
    x, X0, C0, C_rest = _synthetic()
    W = torch.randn(M, N)
    k = rank_for_removal(M, N, 0.4)
    A, B = whitened_factors(W, standard_statistic(C0, C_rest, T), k)
    assert A.shape == (M, k) and B.shape == (k, N)
    # whitened truncation beats plain SVD truncation on the calibration inputs
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    plain = (U[:, :k] * S[:k]) @ Vh[:k]
    rest = x[:, 1:].reshape(-1, N)
    err_white = ((W - (A @ B).float()) @ x.reshape(-1, N).T).norm()
    err_plain = ((W - plain) @ x.reshape(-1, N).T).norm()
    assert err_white <= err_plain * (1 + 1e-3)  # equality only up to the damping


def test_deflation_is_exact_on_the_sink_and_stores_rank_k():
    x, X0, C0, C_rest = _synthetic()
    W = torch.randn(M, N)
    k = rank_for_removal(M, N, 0.4)
    A, B = deflated_factors(W.double(), standard_statistic(C0, C_rest, T), X0, k)
    assert A.shape == (M, k) and B.shape == (k, N)
    u = sink_direction(X0)[:, 0]
    assert torch.allclose(A @ B @ u, W.double() @ u, atol=1e-6)


def test_both_fixes_cut_the_first_position_error():
    x, X0, C0, C_rest = _synthetic()
    rest = x[:, 1:].reshape(-1, N)
    W = torch.randn(M, N)
    k = rank_for_removal(M, N, 0.4)
    e_std = _errors(W, *whitened_factors(W.double(), standard_statistic(C0, C_rest, T), k), X0, rest)
    e_rw = _errors(W, *whitened_factors(W.double(), reweighted_statistic(C0, C_rest, 0.3), k), X0, rest)
    e_df = _errors(W, *deflated_factors(W.double(), standard_statistic(C0, C_rest, T), X0, k), X0, rest)
    # position-0 inputs are the sink direction plus a little noise, so the error there drops by a large factor
    # (deflation is exact on the sink direction itself, see test_deflation_is_exact_on_the_sink_and_stores_rank_k)
    assert e_rw[0] < 0.5 * e_std[0]
    assert e_df[0] < 0.1 * e_std[0]
    # ... at a similar error on the other positions (one rank is spent on the sink)
    assert abs(e_rw[1] - e_std[1]) < 0.1 and abs(e_df[1] - e_std[1]) < 0.1
