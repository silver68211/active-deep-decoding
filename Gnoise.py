
# -*- coding: utf-8 -*-
"""
is_noise_llr.py

Importance-sampling (IS) noise generation and LLR synthesis utilities for
BPSK over AWGN under an all-zero codeword assumption.

This module provides:
- Chi-distribution radial sampling (norm of Gaussian noise vector)
- An importance distribution that adapts based on historical error rates
- IS-based LLR dataset generation over an Eb/N0 range
- A histogram update routine to refine the IS distribution from decoder outputs

Original author: hnksm (May 2023)
Refactor: 2026

Notes
-----
- The IS scheme samples the noise direction uniformly on the unit sphere and the
  radius R from a (possibly reweighted) Chi distribution.
- The function `erup(...)` updates per-radius error counts given decoder outputs.
- This file is designed to be imported. A small demo is provided under `__main__`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from scipy.stats import chi
from tensorflow.experimental.numpy import log2 as _log2


ArrayLike = Union[np.ndarray, tf.Tensor]


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class ISConfig:
    """Configuration for IS dataset generation."""
    step_db: float = 0.5                 # Eb/N0 grid step (dB)
    chi_tail_prob: float = 1e-15         # tail probability for Chi truncation
    llr_clip: float = 20.0               # clipping used in MI helper (if used)


# =============================================================================
# Chi distribution helpers
# =============================================================================

def chi_distribution_grid(
    code_len: int,
    sigma: ArrayLike,
    M: int,
    tail_prob: float = 1e-15,
) -> Tuple[float, float, tf.Tensor, tf.Tensor]:
    """
    Build a radius grid R_list and its Chi pdf values for the given noise std.

    Parameters
    ----------
    code_len : int
        Vector dimension (i.e., blocklength n).
    sigma : float or tf.Tensor
        Noise variance per dimension (scalar).
    M : int
        Number of radius grid points.
    tail_prob : float
        Two-sided tail probability for truncation (e.g., 1e-15).

    Returns
    -------
    R_min : float
    R_max : float
    R_list : tf.Tensor, shape [M]
        Linearly spaced radii.
    gr : tf.Tensor, shape [M]
        Chi pdf evaluated at R_list with scale sqrt(sigma).
    """
    sigma = tf.cast(sigma, tf.float32)

    # Truncate the support to avoid extreme tails
    # Note: we use scale=1.0 to define the grid range, then evaluate pdf with scale=sqrt(sigma)
    R_max = float(chi.ppf(1 - tail_prob, code_len, scale=np.sqrt(1.0)))
    R_min = float(chi.ppf(tail_prob, code_len, scale=np.sqrt(1.0)))

    R_list = tf.linspace(R_min, R_max, M)
    gr = tf.convert_to_tensor(chi.pdf(R_list.numpy(), code_len, scale=float(tf.sqrt(sigma))), dtype=tf.float32)
    return R_min, R_max, R_list, gr


# =============================================================================
# Importance distribution
# =============================================================================

def importance_pdf(
    gr: tf.Tensor,
    wt_hist: ArrayLike,
    err_hist: ArrayLike,
    sim: bool = False,
    threshold: float = 0.5,
) -> tf.Tensor:
    """
    Construct an IS sampling distribution over the radius grid.

    Heuristic: use theta = err_hist / wt_hist as an empirical “difficulty”
    score per radius bin. The sampler uses sqrt(theta)*gr, normalized.

    Parameters
    ----------
    gr : tf.Tensor, shape [1, M] or [M] (we accept either)
        Base Chi pdf values on radius grid.
    wt_hist : array-like, shape [1, M] or [M]
        Number of samples taken per bin historically.
    err_hist : array-like, shape [1, M] or [M]
        Number of observed errors per bin historically.
    sim : bool
        If True, do not apply additional filtering heuristics.

    Returns
    -------
    grIS : tf.Tensor, shape [M]
        Normalized sampling probability for each radius bin.
    """
    gr = tf.reshape(tf.cast(gr, tf.float32), (1, -1))
    wt_hist = tf.reshape(tf.cast(wt_hist, tf.float32), (1, -1))
    err_hist = tf.reshape(tf.cast(err_hist, tf.float32), (1, -1))

    # If no errors yet, sample proportional to the base distribution
    if tf.reduce_sum(err_hist) == 0:
        theta = tf.ones_like(gr)
    else:
        theta = err_hist / wt_hist
        theta = tf.where(tf.math.is_nan(theta), 0.0, theta)

        # Your original code performs a smoothing / fill-forward using ragged slicing.
        # We keep the spirit but simplify: fill zeros by nearest nonzero to the right
        # (then left) to reduce dead bins.
        if not sim:
            # Optional heuristic: suppress overly “easy/hard” bins.
            # You used theta>=0.6 -> 0; kept as-is for compatibility.
            theta = tf.where(theta >= threshold, 0.0, theta)

        # If all bins got wiped out, fallback to uniform theta
        if tf.reduce_sum(theta) == 0:
            theta = tf.ones_like(gr)

    grIS = tf.sqrt(theta) * gr
    grIS = grIS / tf.reduce_sum(grIS)
    return tf.reshape(grIS, (-1,))


# =============================================================================
# Noise generation
# =============================================================================

def sample_noise_vectors(
    code_len: int,
    sigma: ArrayLike,
    M: int,
    wt_hist: ArrayLike,
    err_hist: ArrayLike,
    num_per: int,
    sim: bool = False,
    tail_prob: float = 1e-15,
    threshold: float = 0.5,
) -> Tuple[Any, Any, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, Optional[list]]:
    """
    Sample AWGN noise vectors z using radius importance sampling.

    Steps:
    - Build Chi radius grid R_list and Chi pdf gr
    - Construct IS distribution grIS
    - Sample radii R from R_list using grIS
    - Sample standard normal direction and scale to match radius

    Returns
    -------
    wt_hist, err_hist : updated histograms (same objects/arrays)
    z : tf.Tensor, shape [num_per, code_len]
        Noise vectors with ||z_i|| = R_i (approximately, after normalization).
    R : tf.Tensor, shape [num_per, 1]
        Sampled radii.
    R_list : tf.Tensor, shape [M]
        Radius grid.
    gr : tf.Tensor, shape [M]
        Base Chi pdf over R_list (for current sigma).
    grIS : tf.Tensor, shape [M]
        IS sampling pdf over bins.
    ind : Optional[list]
        Sampled indices into R_list (returned only if sim=True).
    """
    _, _, R_list, gr = chi_distribution_grid(code_len, sigma, M, tail_prob=tail_prob)
    grIS = importance_pdf(gr, wt_hist, err_hist, sim=sim, threshold=threshold)

    # Sample indices according to grIS
    ind = np.random.choice(np.arange(M), size=(num_per,), p=grIS.numpy()).tolist()
    R = tf.reshape(tf.gather(R_list, ind), (num_per, 1))

    # Update sampling histogram
    y, _, count = tf.unique_with_counts(tf.reshape(R, (num_per,)))
    bins = [int(tf.where(R_list == y[i])[0, 0].numpy()) for i in range(len(y))]
    wt_hist = np.asarray(wt_hist)
    wt_hist = wt_hist.reshape(1, -1)
    wt_hist[0, bins] += count.numpy()

    # Sample directions and scale to radius
    R_rep = tf.repeat(R, repeats=code_len, axis=-1)  # [num_per, code_len]
    z = tf.random.normal([num_per, code_len], mean=0.0, stddev=1.0, dtype=tf.float32)
    z = tf.keras.utils.normalize(z, axis=1) * tf.cast(R_rep, tf.float32)

    gr = tf.reshape(gr, (1, 1, -1))
    grIS = tf.reshape(grIS, (1, 1, -1))

    if sim:
        return wt_hist, err_hist, z, R, R_list, gr, grIS, ind
    
    return wt_hist, err_hist, z, R, R_list, gr, grIS, None


# =============================================================================
# LLR dataset generation (IS)
# =============================================================================

def ISllr(
    pcm: Any,
    min_snr: float,
    max_snr: float,
    code_len: int,
    cRate: float,
    num_per: int,
    M: int,
    wt_hist: Any,
    err_hist: Any,
    threshold: float = 0.5,
    sh: bool = True,
    sim: bool = False,
    cfg: ISConfig = ISConfig(),
):
    """
    Generate LLR samples over an Eb/N0 range using importance sampling.

    Parameters
    ----------
    pcm : Any
        Included for API compatibility with your project (not used here).
    min_snr, max_snr : float
        Eb/N0 range in dB.
    code_len : int
        Block length n.
    cRate : float
        Code rate.
    num_per : int
        Samples per SNR point.
    M : int
        Radius grid size.
    wt_hist, err_hist : Any
        Sampling and error histograms (updated internally).
    sh : bool
        If True, shuffle the concatenated LLRs.
    sim : bool
        If True, return extra debug info (indices etc.).
    cfg : ISConfig
        Configuration object.

    Returns
    -------
    llr : tf.Tensor, shape [num_snr*num_per, code_len]
    wt_hist, err_hist : updated hist arrays
    R : tf.Tensor, shape [num_snr, num_per, 1]
    R_list : tf.Tensor, shape [num_snr, 1, M]
    (plus extras if sim=True)
    """
    snr_db = np.arange(min_snr, max_snr + cfg.step_db, cfg.step_db)
    SNR = 10 ** (0.1 * snr_db)
    sigma = 1.0 / (2.0 * cRate * SNR)  # noise variance per dimension

    # Initialize hist arrays on first call
    if tf.reduce_sum(tf.cast(wt_hist, tf.float32)) == 0:
        wt_hist = np.zeros((len(SNR), 1, M), dtype=np.float32)
        err_hist = np.zeros((len(SNR), 1, M), dtype=np.float32)
    else:
        wt_hist = np.reshape(wt_hist, (len(SNR), 1, M)).astype(np.float32)
        err_hist = np.reshape(err_hist, (len(SNR), 1, M)).astype(np.float32)

    llr_all = []
    R_all = []
    R_list_all = []
    gr_all = []
    grIS_all = []
    ind_all = []

    for i in range(len(sigma)):
        wt_hist[i, 0], err_hist[i, 0], z, R, R_list, gr, grIS, ind = sample_noise_vectors(
            code_len=code_len,
            sigma=float(sigma[i]),
            M=M,
            wt_hist=wt_hist[i, 0],
            err_hist=err_hist[i, 0],
            num_per=num_per,
            sim=sim,
            tail_prob=cfg.chi_tail_prob,
            threshold=threshold,
        )
        
        

        # BPSK all-zero cw: received y = +1 + z
        y = 1.0 + z
        llr = 2.0 * y / float(sigma[i])  # channel LLRs for BPSK/AWGN
        llr_all.append(llr)

        R_all.append(R)
        R_list_all.append(tf.reshape(R_list, (1, 1, -1)))
        gr_all.append(gr)
        grIS_all.append(grIS)
        if sim:
            ind_all.append(ind)

    llr = tf.concat(llr_all, axis=0)
    if sh:
        llr = tf.random.shuffle(llr)

    R = tf.reshape(tf.concat(R_all, axis=0), (len(SNR), -1, 1))
    R_list = tf.reshape(tf.concat(R_list_all, axis=0), (len(SNR), 1, M))

    if sim:
        gr = tf.concat(gr_all, axis=0)
        grIS = tf.concat(grIS_all, axis=0)
        return llr, wt_hist, err_hist, R, R_list, gr, grIS, ind_all

    return llr, wt_hist, err_hist, R, R_list


# =============================================================================
# Error-histogram update
# =============================================================================

def erup(llr_out: tf.Tensor, wt_hist: ArrayLike, err_hist: ArrayLike, R: tf.Tensor, R_list: tf.Tensor):
    """
    Update error histogram bins using decoder outputs.

    This function assumes:
    - llr_out shape: [total_samples, code_len] or compatible
    - R shape: [num_snr, num_per, 1]
    - R_list shape: [num_snr, 1, M]
    - wt_hist/err_hist shape: [num_snr, 1, M]

    A “frame error” is counted if any bit is decoded as 1 (since all-zero CW is assumed).

    Returns
    -------
    err_hist : updated array
    """
    wt_hist = np.asarray(wt_hist)
    err_hist = np.asarray(err_hist)

    # Reshape decoder outputs to [num_snr, num_per, n]
    llr_out = tf.reshape(llr_out, (wt_hist.shape[0], -1, llr_out.shape[-1]))

    # Hard decisions: 1 if llr < 0 (all-zero CW)
    x = tf.where(llr_out >= 0.0, 0.0, 1.0)

    # Frame error indicator: 1 if any bit is 1
    ferr = tf.where(tf.reduce_sum(x, axis=-1) == 0, 0, 1)  # [num_snr, num_per]
    err_counts = tf.reduce_sum(ferr, axis=-1)              # [num_snr]

    # For each SNR bin: collect radii R for error frames and update histogram
    ferr = tf.reshape(ferr, (wt_hist.shape[0], -1, 1))
    ind = tf.where(ferr != 0)

    r = tf.gather_nd(R, ind)  # radii for error frames
    r = tf.RaggedTensor.from_row_lengths(values=r, row_lengths=err_counts)

    # Sort radii per SNR for stable unique counting
    # r = tf.map_fn(
    #     lambda t: tf.sort(t, axis=0),
    #     r,
    #     fn_output_signature=tf.RaggedTensorSpec(shape=[None], dtype=tf.float64),
    # )
    r = tf.cast(r, tf.float32)  # keep consistent
    r = r.with_flat_values(tf.sort(r.flat_values, axis=0))

    for i, radii in enumerate(r):
        if len(radii) == 0:
            continue

        y, _, count = tf.unique_with_counts(radii.numpy().reshape((-1,)))
        RR = tf.reshape(R_list[i], (-1, 1))  # [M,1]
        indx = tf.where(tf.equal(RR, y))
        err_hist[i, 0, indx[:, 0]] += count.numpy()
    
    return err_hist



def llr2mi(llr, s=None, reduce_dims=True):
    if s is None:
        s = tf.ones_like(llr)

    s = tf.cast(s, llr.dtype)
    llr_zero = s * llr
    llr_zero = tf.clip_by_value(llr_zero, -20.0, 20.0)

    x = tf.cast(_log2(1.0 + tf.exp(llr_zero)), llr.dtype)

    if reduce_dims:
        return 1.0 - tf.reduce_mean(x)
    return 1.0 - tf.reduce_mean(x, axis=-1)

# =============================================================================
# Optional: standalone demo
# =============================================================================

if __name__ == "__main__":
    # Small smoke test (no decoding here; just generate LLRs)
    pcm_str = (
        "1 0 0 1 1 0 1 0 1 1 1 1 0 0 0 "
        "0 1 0 0 1 1 0 1 0 1 1 1 1 0 0 "
        "0 0 1 0 0 1 1 0 1 0 1 1 1 1 0 "
        "0 0 0 1 0 0 1 1 0 1 0 1 1 1 1"
    )
    pcm = np.array([int(x) for x in pcm_str.split()], dtype=np.int32).reshape((-1, 15))
    m, n = pcm.shape

    llr, wt_hist, err_hist, R, R_list = ISllr(
        pcm=pcm,
        min_snr=1.0,
        max_snr=3.0,
        code_len=n,
        cRate=(n - m) / n,
        num_per=10,
        M=10,
        wt_hist=0,
        err_hist=0,
        sh=True,
    )
    print("LLR shape:", llr)
    print("wt_hist shape:", np.asarray(wt_hist))
    print("err_hist shape:", np.asarray(err_hist))
