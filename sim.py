# -*- coding: utf-8 -*-
"""
simulate_bp_vs_model.py

Monte-Carlo simulation script for comparing:
1) A learned/weighted BP model (loaded as a Keras model) vs.
2) A baseline BP decoder (LDPCBPDecoder) with a fixed number of iterations.

The simulation assumes:
- BPSK modulation
- All-zero codeword transmission (x = +1 vector)
- AWGN channel

Outputs
-------
- BER and FER curves saved as CSV
- BER and FER plots saved as PNG
- Optional PCM visualization saved as PNG

Original author: hnksm (May 2023)
Refactor: 2026

Notes
-----
- Your original script had many unused imports and repeated PCM assignments.
- This refactor keeps the logic but makes it readable, configurable, and safer.
- You MUST load/provide `model` (your trained neural/weighted BP model) before running.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tkinter import N
from typing import List, Tuple
import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.io import loadmat

from BPSio import LDPCBPDecoder
from Alist2bin import Alist2bin

# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class SimConfig:
    # Eb/N0 sweep
    ebn0_min_db: float = 4.0
    ebn0_max_db: float = 5.5
    ebn0_step_db: float = 0.5

    # Simulation control
    batch_size: int = 1000
    bp_iters: int = 5

    # Stopping criteria (match your original style)
    stop_bit_errors: int = 100
    stop_frame_errors: int = 100
    stop_frame_errors_high_snr: int = 100
    high_snr_threshold_db: float = 8.8
    hard_cap_samples: int = int(1e10)

    # IO
    os.makedirs("figs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    figs_dir: str = "figs"
    out_dir: str = "results"  # where to save results


# =============================================================================
# Utility: PCM loading + visualization
# =============================================================================

def load_pcm_from_alist(alist_path: str) -> np.ndarray:
    """Load parity-check matrix H from an ALIST file."""
    pcm = Alist2bin(alist_path).astype(np.int32)
    return pcm


def save_pcm_image(pcm: np.ndarray, out_path: str) -> None:
    """Save a simple visualization of the PCM."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(pcm, cmap="binary", aspect="auto")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# =============================================================================
# Baseline BP decoder wrapper
# =============================================================================

def bp_decode_logits(
    pcm: np.ndarray,
    llr: tf.Tensor,
    num_iter: int,
    cn_type: str = "boxplus",
) -> tf.Tensor:
    """
    Run baseline BP decoding using LDPCBPDecoder.

    Parameters
    ----------
    pcm : np.ndarray
        Parity-check matrix (m x n).
    llr : tf.Tensor
        Channel LLRs, shape [batch, n].
    num_iter : int
        Number of BP iterations.
    cn_type : str
        CN update type used by decoder.

    Returns
    -------
    c_hat : tf.Tensor
        Soft outputs (logits-like), shape [batch, n].
    """
    decoder = LDPCBPDecoder(
        pcm,
        num_iter=1,          # we unroll iterations manually to keep state
        stateful=True,
        hard_out=False,
        cn_type=cn_type,
    )
    msg_vn = None
    c_hat = None
    for _ in range(num_iter):
        c_hat, msg_vn = decoder([llr, msg_vn])
    return c_hat


# =============================================================================
# Channel: BPSK AWGN, all-zero CW
# =============================================================================

def awgn_llr_allzero(batch_size: int, n: int, sigma: float) -> tf.Tensor:
    """
    Generate LLRs for BPSK over AWGN when the all-zero codeword is transmitted.

    All-zero bits -> BPSK symbols +1.
    y = 1 + sqrt(sigma) * z , z~N(0,1)
    LLR = 2y/sigma
    """
    y = 1.0 + tf.cast(tf.sqrt(sigma), tf.float32) * tf.random.normal((batch_size, n), mean=0.0, stddev=1.0)
    llr = 2.0 * y / sigma
    return llr


# =============================================================================
# Metrics
# =============================================================================

def ber_fer_from_logits(logits: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Compute bit and frame errors under all-zero codeword assumption.

    logits >= 0 -> decoded 0
    logits <  0 -> decoded 1

    Returns
    -------
    bit_errors : tf.Tensor scalar
    frame_errors : tf.Tensor scalar
    ber : tf.Tensor scalar (per bit)
    fer : tf.Tensor scalar (per frame)
    """
    dec_bits = tf.where(logits >= 0.0, 0, 1)  # [B, n]
    bit_err_per_frame = tf.reduce_sum(tf.cast(dec_bits, tf.float32), axis=-1)  # [B]
    frame_err = tf.where(bit_err_per_frame == 0.0, 0.0, 1.0)

    bit_errors = tf.reduce_sum(bit_err_per_frame)
    frame_errors = tf.reduce_sum(frame_err)

    ber = bit_errors / (tf.cast(tf.size(dec_bits), tf.float32))
    fer = frame_errors / tf.cast(tf.shape(dec_bits)[0], tf.float32)
    return bit_errors, frame_errors, ber, fer


# =============================================================================
# Simulation core
# =============================================================================

def simulate(
    pcm: np.ndarray,
    model: tf.keras.Model,
    cfg: SimConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run Monte-Carlo simulation over Eb/N0 and compare:
    - `model(llr)` outputs (learned BP / ACT-BP etc.)
    - baseline BP decoding (LDPCBPDecoder) with cfg.bp_iters iterations

    Returns
    -------
    BER_model, FER_model, BER_bp, FER_bp : np.ndarray
        Arrays of length = number of Eb/N0 points.
    """
    m, n = pcm.shape
    rate = (n - m) / n

    ebn0_db = np.arange(cfg.ebn0_min_db, cfg.ebn0_max_db + cfg.ebn0_step_db, cfg.ebn0_step_db)
    snr_lin = 10 ** (0.1 * ebn0_db)
    sigma_list = 1.0 / (2.0 * rate * snr_lin)

    BER_model: List[float] = []
    FER_model: List[float] = []
    BER_bp: List[float] = []
    FER_bp: List[float] = []

    # Output directories
    figs_dir = Path(cfg.figs_dir)
    out_dir = Path(cfg.out_dir)
    figs_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, sigma in enumerate(sigma_list):
        print(f"\n{'-'*70} Eb/N0={ebn0_db[i]:.2f} dB {'-'*70}")

        itr = 0
        total_bit_err_model = 0.0
        total_frame_err_model = 0.0
        total_bit_err_bp = 0.0
        total_frame_err_bp = 0.0

        while True:
            itr += 1

            llr = awgn_llr_allzero(cfg.batch_size, n, float(sigma))

            # --- Learned model output ---
            # Your model returns (c, outs, loss). We only need outs.
            model_out = model(llr)
            if isinstance(model_out, (tuple, list)) and len(model_out) >= 2:
                outs = model_out[1]
            else:
                outs = model_out  # fallback if model directly returns logits

            outs = tf.reshape(outs, (cfg.batch_size, n))

            # --- Baseline BP output ---
            outs_bp = bp_decode_logits(pcm, llr, num_iter=cfg.bp_iters, cn_type="boxplus")
            outs_bp = tf.reshape(outs_bp, (cfg.batch_size, n))

            # --- Error metrics ---
            be_m, fe_m, ber_m, fer_m = ber_fer_from_logits(outs)
            be_b, fe_b, ber_b, fer_b = ber_fer_from_logits(outs_bp)

            total_bit_err_model += float(be_m.numpy())
            total_frame_err_model += float(fe_m.numpy())
            total_bit_err_bp += float(be_b.numpy())
            total_frame_err_bp += float(fe_b.numpy())

            # Running estimates
            samples = cfg.batch_size * itr
            ber_run_m = total_bit_err_model / (samples * n)
            fer_run_m = total_frame_err_model / samples
            ber_run_b = total_bit_err_bp / (samples * n)
            fer_run_b = total_frame_err_bp / samples

            print(
                f"Eb/N0 {ebn0_db[i]:.2f} | "
                f"Model BER {ber_run_m:1.6e} FER {fer_run_m:1.6e} | "
                f"BP({cfg.bp_iters}) BER {ber_run_b:1.6e} FER {fer_run_b:1.6e} | "
                f"bitErr {total_bit_err_model:3.0f}/{total_bit_err_bp:3.0f} "
                f"frameErr {total_frame_err_model:3.0f}/{total_frame_err_bp:3.0f} "
                f"samples {samples:10d}"
            )

            # --- Stopping conditions (kept close to your original intent) ---
            if (total_bit_err_model >= cfg.stop_bit_errors) and (total_frame_err_model >= cfg.stop_frame_errors):
                break

            if (
                (total_bit_err_model >= cfg.stop_bit_errors)
                and (total_frame_err_model >= cfg.stop_frame_errors_high_snr)
                and (ebn0_db[i] >= cfg.high_snr_threshold_db)
            ):
                break

            if samples >= cfg.hard_cap_samples:
                break

        BER_model.append(ber_run_m)
        FER_model.append(fer_run_m)
        BER_bp.append(ber_run_b)
        FER_bp.append(fer_run_b)

        # Save intermediate CSVs (so you don’t lose results on interruption)
        np.savetxt(out_dir / f"BER_model_N{n}_K{n-m}.csv", np.array(BER_model), delimiter=",")
        np.savetxt(out_dir / f"FER_model_N{n}_K{n-m}.csv", np.array(FER_model), delimiter=",")
        np.savetxt(out_dir / f"BER_bp_N{n}_K{n-m}.csv", np.array(BER_bp), delimiter=",")
        np.savetxt(out_dir / f"FER_bp_N{n}_K{n-m}.csv", np.array(FER_bp), delimiter=",")

    return np.array(BER_model), np.array(FER_model), np.array(BER_bp), np.array(FER_bp)


# =============================================================================
# Plotting
# =============================================================================

def semilogy_curve(x: np.ndarray, y: np.ndarray, label: str, title: str, save_path: str) -> None:
    """Save a semilogy plot (publication-friendly defaults)."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 7))
    plt.semilogy(x, y, marker="o", linewidth=2, label=label)
    plt.xlabel(r"$E_b/N_0$ (dB)", fontsize=13)
    plt.ylabel(title, fontsize=13)
    plt.grid(True, which="both", linestyle="--", linewidth=0.6)
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    
    


# =============================================================================
# Main
# =============================================================================

def main():
    cfg = SimConfig()

    # -----------------------
    # Load PCM (choose ONE)
    # -----------------------
    pcm = load_pcm_from_alist("codes/BCH_7_4_1_strip.alist.txt")
    
    save_pcm_image(pcm, f"{cfg.figs_dir}/pcm_N{pcm.shape[1]}.png")

    m, n = pcm.shape

    # -----------------------
    # Load your trained model
    # -----------------------
    # IMPORTANT: Provide the correct path and custom_objects if needed.
    # Example:
    model = tf.keras.models.load_model(f"models/BCHR_ISmodel1_N{n}_K{n-m}",compile=False,
                                      custom_objects={"LDPCBPDecoder": LDPCBPDecoder})

    # -----------------------
    # Run simulation
    # -----------------------
    BER_m, FER_m, BER_b, FER_b = simulate(pcm, model, cfg)

    


if __name__ == "__main__":
    main()
