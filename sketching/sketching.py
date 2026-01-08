"""
Sketching / CUR-style matrix approximation experiments.

This module is a cleaned-up version of the original exploratory notebook
`main_sketching.ipynb`. It provides:

- Basis generators for side-information matrices S (polynomial / DCT)
- Synthetic data generation for a QS (+ noise) matrix model
- Column sampling utilities and sampling operator construction
- A small set of sketching helpers (CountSketch / SRFT / real FFT reparameterization)
- Experiment runners that return NMSE and timing

The code is intentionally lightweight and dependency-minimal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple

import time
import random

import numpy as np
from numpy.linalg import svd, pinv, norm


BasisKind = Literal["poly", "dct"]
RowSampling = Literal["random", "active", "real", "combination"]


# -----------------------------
# Basis / sketch transforms
# -----------------------------

def get_S(l: int, m: int, x: Sequence[float], kind: BasisKind = "dct") -> np.ndarray:
    """
    Construct a side-information basis matrix S of shape (l, m).

    Parameters
    ----------
    l : int
        Number of basis rows (modes).
    m : int
        Signal length / number of columns.
    x : sequence of float
        Grid values; only used for polynomial basis.
    kind : {"poly","dct"}
        Basis type.

    Notes
    -----
    - "poly": row i is x^(l-1-i).
    - "dct": simple DCT-like cosine basis used in the original notebook.
    """
    if l <= 0 or m <= 0:
        raise ValueError("l and m must be positive integers.")
    if kind not in ("poly", "dct"):
        raise ValueError("kind must be one of {'poly','dct'}.")

    S = np.zeros((l, m), dtype=float)

    if kind == "poly":
        if len(x) != m:
            raise ValueError("For 'poly' basis, len(x) must equal m.")
        x_arr = np.asarray(x, dtype=float)
        # power decreases down the rows (same convention as the notebook)
        for i in range(l):
            S[i, :] = x_arr ** (l - i)
    else:  # "dct"
        # Match the notebook's DCT-like construction:
        # S[i,j] = cos(pi*(j+1)*(0.5+(i+1))/100)
        jj = np.arange(1, m + 1, dtype=float)[None, :]
        ii = np.arange(1, l + 1, dtype=float)[:, None]
        S = np.cos((np.pi * jj * (0.5 + ii)) / 100.0)

    # Small random column-wise perturbation (kept to preserve original behavior)
    # In the notebook: S[:,j] = S[:,0] + 0.001*rand(...)
    # That makes all columns near-identical; likely accidental but we keep it optional.
    # Comment out if you want a "clean" basis.
    for j in range(m):
        S[:, j] = S[:, 0] + 0.001 * np.random.random(S[:, 0].shape)

    return S


def get_transS(l: int, n: int, x: Sequence[float]) -> np.ndarray:
    """
    Transposed polynomial basis used in the SQ variant in the notebook.

    Returns S of shape (n, l) with S[i,j] = x[i]^j.
    """
    if l <= 0 or n <= 0:
        raise ValueError("l and n must be positive integers.")
    if len(x) != n:
        raise ValueError("len(x) must equal n.")
    x_arr = np.asarray(x, dtype=float)
    S = np.zeros((n, l), dtype=float)
    for i in range(n):
        for j in range(l):
            S[i, j] = x_arr[i] ** j
    return S


def countSketch(m: int, s: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Basic CountSketch matrix C of shape (s, m).

    Each column has exactly one nonzero entry, +/-1.
    """
    if s <= 0 or m <= 0:
        raise ValueError("s and m must be positive integers.")
    rng = rng or np.random.default_rng()
    C = np.zeros((s, m), dtype=float)
    rows = rng.integers(0, s, size=m)
    signs = rng.choice([-1.0, 1.0], size=m)
    C[rows, np.arange(m)] = signs
    return C


def srft(m: int, s: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    A simple SRFT-like sketch matrix of shape (s, m) for real inputs.

    This is a *practical* SRFT: D * F * P, where
    - D is a random +/-1 diagonal
    - F is the unitary DFT matrix (implemented implicitly via FFT)
    - P selects s rows uniformly at random.

    Returns an explicit dense matrix for convenience (small m).
    """
    rng = rng or np.random.default_rng()
    D = rng.choice([-1.0, 1.0], size=m)
    # Build explicit SRFT by applying to standard basis.
    # For small problems this is fine; for large m, apply implicitly.
    F = np.fft.fft(np.eye(m), axis=0) / np.sqrt(m)
    F = (D[None, :] * F)  # left-multiply by diagonal D
    idx = rng.choice(m, size=s, replace=False)
    return F[idx, :].real  # keep real part for real experiments


def realfft_row(A: np.ndarray) -> np.ndarray:
    """
    Reparameterize per-row FFT coefficients into a real vector (matches notebook helper).
    """
    n_int = A.shape[1]
    fft_mat = np.fft.fft(A, axis=1) / np.sqrt(n_int)

    if n_int % 2 == 1:
        cutoff = (n_int + 1) // 2
        idx_real = list(range(1, cutoff))
        idx_imag = list(range(cutoff, n_int))
    else:
        cutoff = n_int // 2
        idx_real = list(range(1, cutoff))
        idx_imag = list(range(cutoff + 1, n_int))

    C = fft_mat.real.copy()
    C[:, idx_real] *= np.sqrt(2)
    C[:, idx_imag] = fft_mat[:, idx_imag].imag * np.sqrt(2)
    return C


def realfft_col(A: np.ndarray) -> np.ndarray:
    """Column-wise variant of `realfft_row`."""
    return realfft_row(A.T).T


# -----------------------------
# Synthetic data + sampling
# -----------------------------

@dataclass
class QSData:
    X_true: np.ndarray
    Q_true: np.ndarray
    S: np.ndarray
    QS: np.ndarray
    E: np.ndarray
    E_perb_component: np.ndarray
    E_perb_norm: float


def generate_qs_matrix(
    n: int,
    l: int,
    S: np.ndarray,
    sparsity: float = 0.0,
    noise_scale: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> QSData:
    """
    Generate a synthetic matrix X = Q S + E (QS model).

    This mirrors the logic from `true_X_QS` in the notebook.

    Parameters
    ----------
    n : int
        Number of rows of X (and Q).
    l : int
        Latent dimension (rows of S, columns of Q).
    S : ndarray, shape (l, m)
        Side-information basis.
    sparsity : float in [0,1]
        Fraction of entries in Q set to zero (randomly chosen).
    noise_scale : float
        Scale of the "orthogonal" noise component (E_perb in the notebook).
    rng : np.random.Generator, optional
        RNG for reproducibility.

    Returns
    -------
    QSData
    """
    rng = rng or np.random.default_rng()
    m = S.shape[1]

    Q = rng.random((n, l))

    if sparsity > 0:
        if not (0 <= sparsity <= 1):
            raise ValueError("sparsity must be in [0, 1].")
        nn = int(np.ceil(n * sparsity))
        ll = int(np.ceil(l * sparsity))
        n_ind = rng.choice(n, size=nn, replace=False)
        l_ind = rng.choice(l, size=ll, replace=False)
        Q[np.ix_(n_ind, l_ind)] = 0.0

    QS = Q @ S

    # Decompose QS to build a structured noise similar to the notebook.
    U, _, Vt = svd(QS, full_matrices=False)
    U1 = U[:, :l]
    V1 = Vt[:l, :]
    U2 = U[:, l:]
    V2 = Vt[l:, :]

    R = rng.random((U1.shape[1], V1.shape[0]))
    R1 = rng.random((U1.shape[1], V1.shape[0]))
    R2 = noise_scale * rng.random((U2.shape[1], V2.shape[0])) if U2.size and V2.size else np.zeros((0, 0))
    E_perb = (U2 @ R2 @ V2) if R2.size else np.zeros_like(QS)

    QS_tilt = U1 @ R @ V1
    E = (U1 @ R1 @ V1) + E_perb

    X_true = QS_tilt + E
    e_perb_norm = float(norm(E_perb))

    return QSData(
        X_true=X_true,
        Q_true=Q,
        S=S,
        QS=QS_tilt,
        E=E,
        E_perb_component=E_perb,
        E_perb_norm=e_perb_norm,
    )


def sample_columns(
    X: np.ndarray,
    d: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample d columns uniformly at random.

    Returns (col_idx, A) where A = X[:, col_idx].
    """
    rng = rng or np.random.default_rng()
    m = X.shape[1]
    if not (1 <= d <= m):
        raise ValueError("d must satisfy 1 <= d <= number of columns.")
    col_idx = rng.choice(m, size=d, replace=False)
    A = X[:, col_idx]
    return col_idx, A


def sampling_matrix(m: int, col_idx: Sequence[int]) -> np.ndarray:
    """
    Build Psi of shape (m, d) such that X @ Psi = X[:, col_idx].
    """
    col_idx = np.asarray(col_idx, dtype=int)
    d = len(col_idx)
    Psi = np.zeros((m, d), dtype=float)
    Psi[col_idx, np.arange(d)] = 1.0
    return Psi


# -----------------------------
# Metrics + experiment runners
# -----------------------------

def nmse(X_hat: np.ndarray, X_true: np.ndarray, eps: float = 1e-12) -> float:
    """Normalized MSE = ||X_hat - X_true||_F^2 / (||X_true||_F^2 + eps)."""
    return float(norm(X_hat - X_true, "fro") ** 2 / (norm(X_true, "fro") ** 2 + eps))


def cur_from_column_samples(A: np.ndarray, B: np.ndarray, col_idx: Sequence[int]) -> np.ndarray:
    """
    Notebook-style CUR reconstruction:
        Z = pinv(A) A pinv(B[:, col_idx])
        M_hat = A Z B
    """
    col_idx = np.asarray(col_idx, dtype=int)
    Z = pinv(A) @ A @ pinv(B[:, col_idx])
    return A @ Z @ B


def run_qs_cur_experiment(
    n: int,
    m: int,
    l: int,
    d: int,
    s: int,
    basis: BasisKind = "dct",
    sparsity: float = 0.0,
    noise_scale: float = 0.0,
    row_sampling: RowSampling = "random",
    seed: int = 42,
) -> Tuple[float, float]:
    """
    End-to-end QS -> column sample -> (random) row sample -> CUR reconstruction.

    Returns
    -------
    (nmse_value, elapsed_seconds)
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    x = np.arange(m, dtype=float)
    S = get_S(l, m, x, kind=basis)

    data = generate_qs_matrix(n=n, l=l, S=S, sparsity=sparsity, noise_scale=noise_scale, rng=rng)

    t0 = time.time()
    col_idx, A = sample_columns(data.X_true, d=d, rng=rng)
    B = (data.QS)[rng.choice(n, size=s, replace=False), :] if row_sampling == "random" else data.X_true[rng.choice(n, size=s, replace=False), :]
    X_hat = cur_from_column_samples(A=A, B=B, col_idx=col_idx)
    elapsed = time.time() - t0

    return nmse(X_hat, data.X_true), elapsed
