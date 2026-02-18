# -*- coding: utf-8 -*-
"""
Weighted Belief Propagation (WBP) training script for linear block codes (e.g., BCH/LDPC).

What this script does
- Loads a parity-check matrix (PCM) from an ALIST file (or uses a hardcoded PCM if desired).
- Builds a Keras model that runs BP decoding for `num_iter` iterations.
- Uses a multi-loss objective (loss after each BP iteration, averaged), following Nachmani et al.-style training.
- Trains with gradient clipping and logs losses to TensorBoard.
- (Optionally) updates importance-sampling (IS) statistics via `erup(...)` if you use IS-based LLR generation.

Dependencies (project-local)
- Alist2bin.py: provides Alist2bin(path) -> np.ndarray
- Gnoise.py: provides ISllr(...) / gllr(...) and erup(...)
- BPSio.py: provides LDPCBPDecoder

Notes
- This is a cleaned, professionalized version of your original script:
  * removed unused/redundant imports
  * removed duplicate `data(...)` function definitions (kept one, IS-style)
  * moved magic numbers into a config dataclass
  * added comments, type hints, and basic input validation
  * kept your training logic and API calls intact
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple, Any
import os
import numpy as np
import tensorflow as tf

from Alist2bin import Alist2bin
from BPSio import LDPCBPDecoder
from Gnoise import ISllr, erup  # keep gllr import if you want non-IS generation


# -----------------------------
# Configuration
# -----------------------------
@dataclass(frozen=True)
class TrainConfig:
    # Data / channel settings
    min_snr_db: float = 7.0
    max_snr_db: float = 8.0
    num_per: int = 20 * 1000
    batch_size: int = 1000
    threshold: float = 0.4

    # Model / decoder settings
    num_iter: int = 5  # BP iterations (and number of losses)
    layer_llr: bool = False  # currently unused by model, kept for compatibility

    # Training settings
    epoch_blocks: Tuple[int, ...] = tuple([5]*1000)  # you had multiple epoch schedules; keep one
    learning_rate: float = 0.01
    clip_value_grad: float = 10.0

    # IS settings (used by ISllr + erup)
    M: int = 100  # IS samples/parameter used in your generator
    model_dir: str = "models"
    log_root: str = "logs/gradient_tape"


# -----------------------------
# Model
# -----------------------------
@tf.keras.saving.register_keras_serializable()
class WeightedBP(tf.keras.Model):
    """
    Weighted BP model that unrolls BP decoding for `num_iter` steps.

    - `self.layers_bp[i]` runs exactly one decoding iteration (num_iter=1) and is stateful.
    - We compute a BCE loss against the all-zero codeword after each iteration and average.
    """

    def __init__(self, pcm: np.ndarray, num_iter: int = 5, layer_llr: bool = False):
        super().__init__()
        if pcm.ndim != 2:
            raise ValueError("pcm must be a 2D array of shape (m, n).")
        self.pcm = pcm.astype(np.int32, copy=False)
        self.n = int(self.pcm.shape[1])
        self.num_iter = int(num_iter)
        self.layer_llr = bool(layer_llr)

        # First iteration is frozen; subsequent are trainable (as in your original code).
        self.layers_bp = [
            LDPCBPDecoder(
                self.pcm,
                num_iter=1,
                stateful=True,
                hard_out=False,
                cn_type="boxplus",
                trainable=True,
                trainable_llr_vn=False,
                trainable_llr_fin=False,
                trainable_msg_fin=True,
            )
        ]
        for _ in range(self.num_iter - 1):
            self.layers_bp.append(
                LDPCBPDecoder(
                    self.pcm,
                    num_iter=1,
                    stateful=True,
                    hard_out=False,
                    cn_type="boxplus",
                    trainable=True,
                    trainable_llr_vn=False,
                    trainable_llr_fin=False,
                    trainable_msg_fin=True,
                )
            )

        self._bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    def get_config(self):
        config = super().get_config()
        config.update({
            "pcm": self.pcm.tolist(),      # serializable
            "num_iter": self.num_iter,
            "layer_llr": self.layer_llr,
        })
        return config

    @classmethod
    def from_config(cls, config):
        pcm = np.array(config.pop("pcm"), dtype=np.int32)
        return cls(pcm=pcm, **config)
    def call(self, llr: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Args:
            llr: Tensor shape [batch, n], channel LLRs (float).
        Returns:
            c: all-zero codeword tensor [batch, n]
            c_hat: final soft output (logits-like) [batch, n]
            loss: scalar tensor
        """
        llr = tf.convert_to_tensor(llr, dtype=tf.float32)

        # All-zero codeword (common for training BP decoders)
        c = tf.zeros([tf.shape(llr)[0], self.n], dtype=tf.float32)

        msg_vn = None
        loss = 0.0
        c_hat = None

        # Multi-loss across iterations
        for i in range(self.num_iter):
            c_hat, msg_vn = self.layers_bp[i]([llr, msg_vn])
            loss += self._bce(c, -c_hat)

        loss /= float(self.num_iter)
        return c, c_hat, loss


# -----------------------------
# Data generation
# -----------------------------
def make_dataset_is(
    pcm: np.ndarray,
    min_snr_db: float,
    max_snr_db: float,
    code_len: int,
    code_rate: float,
    num_per: int,
    batch_size: int,
    M: int,
    wt_hist: Any,
    err_hist: Any,
    threshold: float = 0.5,
    generator: Callable[..., Any] = ISllr,
    shuffle: bool = True,
) -> Tuple[tf.data.Dataset, tf.Tensor, Any, Any, Any, Any]:
    """
    Generates training and validation LLRs using an IS-based generator.

    The generator is assumed to return:
        llr, wt_hist, err_hist, R, R_list

    We create:
        - llr_train: tf.data.Dataset (shuffled + batched)
        - llr_val:   tf.Tensor or np.ndarray (kept as-is, because your test_step expects a full batch)
    """
    llr_train, wt_hist, err_hist, R, R_list = generator(
        pcm, min_snr_db, max_snr_db, code_len, code_rate, num_per, M, wt_hist, err_hist, threshold=threshold, sh=shuffle
    )
    llr_val, wt_hist, err_hist, R, R_list = generator(
        pcm, min_snr_db, max_snr_db, code_len, code_rate, num_per, M, wt_hist, err_hist, threshold=threshold, sh=False
    )

    llr_train_ds = tf.data.Dataset.from_tensor_slices(llr_train)
    if shuffle:
        llr_train_ds = llr_train_ds.shuffle(60000)
    llr_train_ds = llr_train_ds.batch(batch_size)

    # Keep validation as a single big tensor/batch to match your original usage
    llr_val = tf.convert_to_tensor(llr_val, dtype=tf.float32)
    return llr_train_ds, llr_val, wt_hist, err_hist, R, R_list


# -----------------------------
# Train / eval steps
# -----------------------------
@tf.function
def train_step(model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer, llr_batch: tf.Tensor, clip: float) -> tf.Tensor:
    with tf.GradientTape() as tape:
        _, _, loss = model(llr_batch, training=True)

    grads = tape.gradient(loss, model.trainable_variables)
    # Clip gradients elementwise (as in your original code)
    grads = [tf.clip_by_value(g, -clip, clip) if g is not None else None for g in grads]
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


@tf.function
def test_step(model: tf.keras.Model, llr_val: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    _, c_hat, loss = model(llr_val, training=False)
    return loss, c_hat


def make_writers(log_root: str) -> Tuple[tf.summary.SummaryWriter, tf.summary.SummaryWriter, tf.summary.SummaryWriter]:
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base = Path(log_root) / current_time
    train_writer = tf.summary.create_file_writer(str(base / "train"))
    test_writer = tf.summary.create_file_writer(str(base / "test"))
    ber_writer = tf.summary.create_file_writer(str(base / "BER"))
    return train_writer, test_writer, ber_writer


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    cfg = TrainConfig()

    # ---- Load PCM ----
    # Choose ONE PCM source. In your original code you overwrote pcm multiple times.
    pcm_path = "codes/BCH_7_4_1_strip.alist.txt"
    pcm = Alist2bin(pcm_path).astype(np.int32)

    m, n = pcm.shape
    code_rate = (n - m) / n

    # ---- Build model ----
    tf.keras.saving.get_custom_objects().clear()
    model = WeightedBP(pcm=pcm, num_iter=cfg.num_iter, layer_llr=cfg.layer_llr)

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=cfg.learning_rate)

    # ---- Logging ----
    train_writer, test_writer, _ = make_writers(cfg.log_root)

    # ---- IS history ----
    wt_hist: Any = 0
    err_hist: Any = 0

    # ---- Initial validation + IS update ----
    llr_train, llr_val, wt_hist, err_hist, R, R_list = make_dataset_is(
        pcm=pcm,
        min_snr_db=cfg.min_snr_db,
        max_snr_db=cfg.max_snr_db,
        code_len=n,
        code_rate=code_rate,
        num_per=cfg.num_per,
        batch_size=cfg.batch_size,
        M=cfg.M,
        wt_hist=wt_hist,
        err_hist=err_hist,
        threshold=cfg.threshold,
        generator=ISllr,
        shuffle=True,
    )
    val_loss, c_hat = test_step(model, llr_val)
    err_hist = erup(llr_out=c_hat, wt_hist=wt_hist, err_hist=err_hist, R=R, R_list=R_list)

    # ---- Training loop ----
    model_dir = Path(cfg.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    global_epoch = 0
    for block_idx, num_epochs in enumerate(cfg.epoch_blocks, start=1):
        # Refresh dataset each block (keeps your original behavior)
        llr_train, llr_val, wt_hist, err_hist, R, R_list = make_dataset_is(
            pcm=pcm,
            min_snr_db=cfg.min_snr_db,
            max_snr_db=cfg.max_snr_db,
            code_len=n,
            code_rate=code_rate,
            num_per=cfg.num_per,
            batch_size=cfg.batch_size,
            M=cfg.M,
            wt_hist=wt_hist,
            err_hist=err_hist,
            generator=ISllr,
            shuffle=True,
        )

        # Evaluate before the block
        val_loss, _ = test_step(model, llr_val)
        num_batches = 0
        for epoch in range(1, num_epochs + 1):
            global_epoch += 1

            # Train
            running_loss = 0.0
            
            for llr_batch in llr_train:
                running_loss += float(train_step(model, optimizer, llr_batch, cfg.clip_value_grad))
                num_batches += 1

            # Validate
            val_loss, _ = test_step(model, llr_val)

            # TensorBoard scalars
            with train_writer.as_default():
                tf.summary.scalar("loss", running_loss, step=global_epoch)
            with test_writer.as_default():
                tf.summary.scalar("loss_val", val_loss, step=global_epoch)

            # Save model (same naming style you used, but deterministic)
            save_path = model_dir / f"BCHR_ISmodel1_N{n}_K{n-m}"
            model.save(str(save_path), save_format="tf")

            print(
                f"[Block {block_idx:03d}/{len(cfg.epoch_blocks):03d}] "
                f"Epoch {epoch:03d}/{num_epochs:03d} | "
                f"Batches {num_batches:04d} | "
                f"TrainLoss {running_loss:1.6e} | ValLoss {float(val_loss):1.6e}"
            )

        # Update IS statistics at the end of each block
        _, c_hat = test_step(model, llr_val)
        err_hist = erup(llr_out=c_hat, wt_hist=wt_hist, err_hist=err_hist, R=R, R_list=R_list)


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    main()
