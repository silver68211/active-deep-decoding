# -*- coding: utf-8 -*-
"""
ldpc_bp_decoder.py

Belief-Propagation (BP) decoder layer for LDPC / linear block codes.

This implementation performs flooding-schedule BP on an arbitrary parity-check
matrix (PCM) using ragged tensors to support irregular degrees.

Key features
-----------
- Supports CN update rules: {"boxplus", "boxplus-phi", "minsum"}
- Optional trainable weights for:
    * VN→CN messages (weighted BP à la Nachmani)
    * final VN messages
    * per-bit channel-LLR scaling (VN stage and/or final stage)
- Optional stateful mode: returns VN messages for iterative detection/decoding
- Optional EXIT tracking (requires all-zero codeword input)

Notes
-----
- Internally, computation is done in tf.float32 for stability.
- Input is assumed to be logits/LLRs; this layer treats them consistently,
  but you should verify sign conventions with your pipeline.

Author: hnksm (original)
Refactor: 2026
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import scipy as sp  # sparse PCM handling
import tensorflow as tf
from tensorflow.keras.layers import Layer

# Project utilities (kept here because you use them in EXIT tracking)
from Gnoise import llr2mi  # gllr, ebnodb2no not used in this file


PcmType = Union[np.ndarray, "sp.sparse.csr_matrix", "sp.sparse.csc_matrix"]


class LDPCBPDecoder(Layer):
    r"""
    Iterative belief propagation decoder layer.

    Parameters
    ----------
    pcm : np.ndarray or scipy sparse matrix
        Parity-check matrix of shape (m, n), entries in {0,1}.
    trainable : bool
        If True, apply per-edge trainable scaling to VN→CN messages.
    trainable_llr_vn : bool
        If True, apply per-bit trainable scaling to channel LLRs at VN update.
    trainable_msg_fin : bool
        If True, apply per-edge trainable scaling to the *final* VN messages.
    trainable_llr_fin : bool
        If True, apply per-bit trainable scaling to channel LLRs at final marginalization.
    cn_type : str
        One of {"boxplus", "boxplus-phi", "minsum"}.
    hard_out : bool
        If True, output hard decisions (0/1). Otherwise output soft values.
    track_exit : bool
        If True, compute mutual information trajectory (requires all-zero CW).
    num_iter : int
        Number of BP iterations.
    stateful : bool
        If True, expects/returns VN messages for iterative processing.
    output_dtype : tf.DType
        Output dtype (internal computations remain tf.float32).
    """

    def __init__(
        self,
        pcm: PcmType,
        trainable: bool = False,
        trainable_llr_vn: bool = False,
        trainable_msg_fin: bool = False,
        trainable_llr_fin: bool = False,
        cn_type: str = "boxplus-phi",
        hard_out: bool = True,
        track_exit: bool = False,
        num_iter: int = 20,
        stateful: bool = False,
        output_dtype: tf.DType = tf.float32,
        **kwargs,
    ):
        super().__init__(dtype=output_dtype, **kwargs)

        # -------------------------
        # Basic argument validation
        # -------------------------
        if not isinstance(trainable, bool):
            raise TypeError("trainable must be bool.")
        if not isinstance(trainable_llr_vn, bool):
            raise TypeError("trainable_llr_vn must be bool.")
        if not isinstance(trainable_msg_fin, bool):
            raise TypeError("trainable_msg_fin must be bool.")
        if not isinstance(trainable_llr_fin, bool):
            raise TypeError("trainable_llr_fin must be bool.")
        if not isinstance(hard_out, bool):
            raise TypeError("hard_out must be bool.")
        if not isinstance(track_exit, bool):
            raise TypeError("track_exit must be bool.")
        if not isinstance(stateful, bool):
            raise TypeError("stateful must be bool.")
        if not isinstance(cn_type, str):
            raise TypeError("cn_type must be str.")
        if not isinstance(num_iter, int) or num_iter < 0:
            raise ValueError("num_iter must be a non-negative int.")
        if not isinstance(output_dtype, tf.DType):
            raise TypeError("output_dtype must be a tf.DType.")

        # -------------------------
        # Store configuration
        # -------------------------
        self._pcm = pcm
        self._trainable_ = trainable
        self._trainable_llr_vn = trainable_llr_vn
        self._trainable_llr_fin = trainable_llr_fin
        self._trainable_msg_fin = trainable_msg_fin
        self._cn_type = cn_type
        self._hard_out = hard_out
        self._track_exit = track_exit
        self._num_iter = tf.constant(num_iter, dtype=tf.int32)
        self._stateful = stateful
        self._output_dtype = output_dtype

        # Numerical stability constants
        self._atanh_clip_value = 1.0 - 1e-7
        self._llr_max_minsum = 10.0

        # -------------------------
        # Prepare sparse PCM + graph structure
        # -------------------------
        if isinstance(pcm, np.ndarray):
            pcm_sp = sp.sparse.csr_matrix(pcm)
        elif isinstance(pcm, (sp.sparse.csr_matrix, sp.sparse.csc_matrix)):
            pcm_sp = pcm.tocsr()
        else:
            raise TypeError("pcm must be np.ndarray or scipy sparse csr/csc matrix.")

        self._num_cns = int(pcm_sp.shape[0])
        self._num_vns = int(pcm_sp.shape[1])

        # Extract all edges (row indices, col indices)
        cn_idx, vn_idx, _ = sp.sparse.find(pcm_sp)
        self._cn_con = cn_idx.astype(np.int32)
        self._vn_con = vn_idx.astype(np.int32)
        self._num_edges = int(len(self._vn_con))

        # Permutation for CN perspective (group edges by CN index)
        self._ind_cn = np.argsort(self._cn_con)
        self._ind_cn_inv = np.argsort(self._ind_cn)

        # Row-splits to build ragged tensors for VN/CN incoming messages
        self._vn_row_splits = self._gen_node_mask_row(self._vn_con)
        self._cn_row_splits = self._gen_node_mask_row(self._cn_con[self._ind_cn])

        # Select CN update implementation
        if self._cn_type == "boxplus":
            self._cn_update = self._cn_update_tanh
        elif self._cn_type == "boxplus-phi":
            self._cn_update = self._cn_update_phi
        elif self._cn_type == "minsum":
            self._cn_update = self._cn_update_minsum
        else:
            raise ValueError("cn_type must be one of {'boxplus','boxplus-phi','minsum'}.")

        # -------------------------
        # Trainable parameters
        # -------------------------
        self._has_weights = False
        self._has_weights_fin = False
        self._has_weights_llr_fin = False
        self._has_weights_llr_vn = False

        if self._trainable_:
            self._has_weights = True
            self._edge_weights = tf.Variable(
                tf.ones([self._num_edges], dtype=tf.float32),
                trainable=True,
                name="edge_weights",
            )

        if self._trainable_msg_fin:
            self._has_weights_fin = True
            self._edge_weights_fin = tf.Variable(
                tf.ones([self._num_edges], dtype=tf.float32),
                trainable=True,
                name="edge_weights_fin",
            )

        if self._trainable_llr_fin:
            self._has_weights_llr_fin = True
            self._llr_weights_fin = tf.Variable(
                tf.ones([self._num_vns], dtype=tf.float32),
                trainable=True,
                name="llr_weights_fin",
            )

        if self._trainable_llr_vn:
            self._has_weights_llr_vn = True
            self._llr_weights_vn = tf.Variable(
                tf.ones([self._num_vns], dtype=tf.float32),
                trainable=True,
                name="llr_weights_vn",
            )

        # EXIT tracking buffers
        self._ie_c = 0.0
        self._ie_v = 0.0

        # CN mask depends only on connectivity; build once as a constant ragged tensor.
        # (This avoids rebuilding it on every call.)
        self._cn_mask_tf = tf.ragged.constant(
            self._gen_node_mask(self._cn_con),
            row_splits_dtype=tf.int32,
        )

    # ---------------------------------------------------------------------
    # Public properties
    # ---------------------------------------------------------------------
    @property
    def pcm(self):
        return self._pcm

    @property
    def num_cns(self) -> int:
        return self._num_cns

    @property
    def num_vns(self) -> int:
        return self._num_vns

    @property
    def num_edges(self) -> int:
        return self._num_edges

    @property
    def output_dtype(self) -> tf.DType:
        return self._output_dtype

    @property
    def ie_c(self):
        return self._ie_c

    @property
    def ie_v(self):
        return self._ie_v

    @property
    def num_iter(self) -> tf.Tensor:
        return self._num_iter

    @num_iter.setter
    def num_iter(self, num_iter: int) -> None:
        if not isinstance(num_iter, int) or num_iter < 0:
            raise ValueError("num_iter must be a non-negative int.")
        self._num_iter = tf.constant(num_iter, dtype=tf.int32)

    # ---------------------------------------------------------------------
    # Utility: masks / ragged helpers
    # ---------------------------------------------------------------------
    def _gen_node_mask(self, con: np.ndarray):
        """Return list-of-lists: incoming edge indices per node (ragged mask)."""
        ind = np.argsort(con)
        con_sorted = con[ind]

        node_mask = []
        cur_node = 0
        cur_mask = []
        for i in range(self._num_edges):
            if con_sorted[i] == cur_node:
                cur_mask.append(int(ind[i]))
            else:
                node_mask.append(cur_mask)
                cur_mask = [int(ind[i])]
                cur_node += 1
        node_mask.append(cur_mask)
        return node_mask

    def _gen_node_mask_row(self, con: np.ndarray):
        """Row-splits for ragged tensor creation from a flat edge-message vector."""
        row_splits = [0]
        cur_node = 0
        for i in range(self._num_edges):
            if con[i] != cur_node:
                row_splits.append(i)
                cur_node += 1
        row_splits.append(self._num_edges)
        return row_splits

    @staticmethod
    def _where_ragged(msg: tf.Tensor) -> tf.Tensor:
        """Replace zeros by small epsilon (avoids division/log issues)."""
        return tf.where(tf.equal(msg, 0.0), tf.ones_like(msg) * 1e-12, msg)

    @staticmethod
    def _where_ragged_inv(msg: tf.Tensor) -> tf.Tensor:
        """Replace tiny values by exact zero (cleanup after epsilon injection)."""
        return tf.where(tf.less(tf.abs(msg), 1e-7), tf.zeros_like(msg), msg)

    @staticmethod
    def _stop_ragged_gradient(rt: tf.RaggedTensor) -> tf.RaggedTensor:
        """Stop gradients for ragged tensor flat values."""
        return rt.with_flat_values(tf.stop_gradient(rt.flat_values))

    # ---------------------------------------------------------------------
    # VN update
    # ---------------------------------------------------------------------
    def _vn_update(self, msg: tf.RaggedTensor, llr_ch: tf.Tensor) -> tf.RaggedTensor:
        """
        Variable node update:
            x_{i->j} = llr_ch[i] + sum_{j' in N(i) \ {j}} y_{j'->i}
        """
        node_sum = tf.reduce_sum(msg, axis=1)          # shape: [num_vns, batch]
        node_sum = tf.add(node_sum, llr_ch)            # add channel LLRs

        # extrinsic: subtract the intrinsic incoming message
        return tf.ragged.map_flat_values(
            lambda x, y, row_ind: x + tf.gather(y, row_ind),
            -1.0 * msg,
            node_sum,
            msg.value_rowids(),
        )

    # ---------------------------------------------------------------------
    # CN update rules
    # ---------------------------------------------------------------------
    def _cn_update_tanh(self, msg: tf.RaggedTensor) -> tf.RaggedTensor:
        """Exact boxplus CN update via tanh/atanh (stable with clipping)."""
        msg = tf.clip_by_value(msg, -self._llr_max_minsum, self._llr_max_minsum)
        msg = msg / 2.0
        msg = tf.ragged.map_flat_values(tf.tanh, msg)
        msg = tf.ragged.map_flat_values(self._where_ragged, msg)

        prod = tf.reduce_prod(msg, axis=1)

        # extrinsic product: divide-out intrinsic term (use power -1 instead of divide)
        msg = tf.ragged.map_flat_values(
            lambda x, y, row_ind: x * tf.gather(y, row_ind),
            msg ** -1,
            prod,
            msg.value_rowids(),
        )

        msg = tf.ragged.map_flat_values(self._where_ragged_inv, msg)
        msg = tf.clip_by_value(msg, -self._atanh_clip_value, self._atanh_clip_value)
        msg = 2.0 * tf.ragged.map_flat_values(tf.atanh, msg)
        msg = tf.clip_by_value(msg, -self._llr_max_minsum, self._llr_max_minsum)
        return msg

    @staticmethod
    def _phi(x: tf.Tensor) -> tf.Tensor:
        """Phi function used in boxplus-phi CN update (Ryan)."""
        x = tf.clip_by_value(x, 8.5e-8, 16.635532)
        return tf.math.log(tf.math.exp(x) + 1.0) - tf.math.log(tf.math.exp(x) - 1.0)

    def _cn_update_phi(self, msg: tf.RaggedTensor) -> tf.RaggedTensor:
        """Exact boxplus CN update via phi function."""
        sign_val = tf.sign(msg)
        sign_val = tf.where(tf.equal(sign_val, 0.0), tf.ones_like(sign_val), sign_val)

        sign_node = tf.reduce_prod(sign_val, axis=1)

        sign_val = tf.ragged.map_flat_values(
            lambda x, y, row_ind: x * tf.gather(y, row_ind),
            sign_val,
            sign_node,
            sign_val.value_rowids(),
        )

        msg_abs = tf.ragged.map_flat_values(tf.abs, msg)
        msg_phi = tf.ragged.map_flat_values(self._phi, msg_abs)
        msg_sum = tf.reduce_sum(msg_phi, axis=1)

        # extrinsic sum
        msg_ext = tf.ragged.map_flat_values(
            lambda x, y, row_ind: x + tf.gather(y, row_ind),
            -1.0 * msg_phi,
            msg_sum,
            msg_phi.value_rowids(),
        )

        msg_out = self._stop_ragged_gradient(sign_val) * tf.ragged.map_flat_values(self._phi, msg_ext)
        return msg_out

    @staticmethod
    def _sign_val_minsum(msg: tf.Tensor) -> tf.Tensor:
        """Sign helper for min-sum decoding (treat 0 as +1)."""
        s = tf.sign(msg)
        return tf.where(tf.equal(s, 0.0), tf.ones_like(s), s)

    def _cn_update_minsum(self, msg: tf.RaggedTensor) -> tf.RaggedTensor:
        """Min-sum CN update with extrinsic minimum."""
        LARGE_VAL = 10000.0

        msg = tf.clip_by_value(msg, -self._llr_max_minsum, self._llr_max_minsum)
        sign_val = tf.ragged.map_flat_values(self._sign_val_minsum, msg)
        sign_node = tf.reduce_prod(sign_val, axis=1)

        sign_val = tf.ragged.map_flat_values(
            lambda x, y, row_ind: tf.multiply(x, tf.gather(y, row_ind)),
            self._stop_ragged_gradient(sign_val),
            sign_node,
            sign_val.value_rowids(),
        )

        msg_abs = tf.ragged.map_flat_values(tf.abs, msg)

        min_val = tf.reduce_min(msg_abs, axis=1, keepdims=True)

        msg_min1 = tf.ragged.map_flat_values(
            lambda x, y, row_ind: x - tf.gather(y, row_ind),
            msg_abs,
            tf.squeeze(min_val, axis=1),
            msg_abs.value_rowids(),
        )

        # ignore min positions for second min
        msg_ign = tf.ragged.map_flat_values(
            lambda x: tf.where(tf.equal(x, 0.0), LARGE_VAL, x),
            msg_min1,
        )

        min_val2 = tf.reduce_min(msg_ign, axis=1, keepdims=True) + min_val

        node_sum = tf.reduce_sum(msg_ign, axis=1, keepdims=True) - (2 * LARGE_VAL - 1.0)
        double_min = 0.5 * (1.0 - tf.sign(node_sum))
        min_val_e = (1.0 - double_min) * min_val + double_min * min_val2

        msg_e = tf.where(msg_ign == LARGE_VAL, min_val_e, min_val)

        # ensure ragged shapes
        msg_e = tf.ragged.map_flat_values(
            lambda x: tf.ensure_shape(x, msg_ign.flat_values.shape),
            msg_e,
        )

        return tf.ragged.map_flat_values(tf.multiply, sign_val, msg_e)

    # ---------------------------------------------------------------------
    # Trainable scaling helper
    # ---------------------------------------------------------------------
    @staticmethod
    def _mult_weights(x: tf.Tensor, w: tf.Tensor) -> tf.Tensor:
        """Multiply messages by weights (broadcast-friendly)."""
        x_t = tf.transpose(x, (1, 0))
        x_t = tf.multiply(x_t, w)
        return tf.transpose(x_t, (1, 0))

    # ---------------------------------------------------------------------
    # Keras call
    # ---------------------------------------------------------------------
    def call(self, inputs):
        """
        Run BP decoding for `num_iter` iterations.

        Input
        -----
        If stateful=False:
            inputs: Tensor [..., n]
        If stateful=True:
            inputs: (llr_ch, msg_vn)
                llr_ch: Tensor [..., n]
                msg_vn : RaggedTensor or None

        Output
        ------
        If stateful=False:
            x_out: Tensor [..., n]
        If stateful=True:
            (x_out, msg_vn_ragged)
        """
        if self._stateful:
            llr_ch, msg_vn = inputs
        else:
            llr_ch = inputs
            msg_vn = None

        # Validate dtype and last dimension
        llr_ch = tf.convert_to_tensor(llr_ch)
        tf.debugging.assert_type(llr_ch, self.dtype, "Invalid input dtype.")
        llr_ch = tf.cast(llr_ch, tf.float32)

        tf.debugging.assert_equal(
            tf.shape(llr_ch)[-1],
            self._num_vns,
            "Last dimension must be of length n.",
        )

        # Flatten any leading dims into one batch dimension for decoding
        llr_shape = llr_ch.get_shape().as_list()
        llr_flat = tf.reshape(llr_ch, [-1, self._num_vns])  # [B, n]

        # For ragged operations, keep batch as last dim: [n, B]
        llr_flat = tf.transpose(llr_flat, (1, 0))

        # Initialize or reuse VN messages
        if (not self._stateful) or (msg_vn is None):
            msg_init_shape = tf.stack([tf.constant(self._num_edges), tf.shape(llr_flat)[1]], axis=0)
            msg_vn_flat = tf.zeros(msg_init_shape, dtype=tf.float32)
        else:
            msg_vn_flat = msg_vn.flat_values

        # EXIT tracking buffers
        if self._track_exit:
            self._ie_c = tf.zeros(self._num_iter + 1, dtype=tf.float32)
            self._ie_v = tf.zeros(self._num_iter + 1, dtype=tf.float32)

        def dec_iter(llr_in, msg_in, it):
            it += 1

            # Optional edge-weight scaling VN→CN
            if self._trainable_:
                msg_in = tf.ragged.map_flat_values(self._mult_weights, msg_in, self._edge_weights)

            # Optional per-bit scaling for channel LLRs at VN update
            if self._trainable_llr_vn:
                llr_vn = tf.ragged.map_flat_values(self._mult_weights, llr_in, self._llr_weights_vn)
            else:
                llr_vn = llr_in

            # Build ragged VN incoming-message tensor: [num_vns, deg(v), B]
            msg_vn_rg = tf.RaggedTensor.from_row_splits(
                values=msg_in,
                row_splits=tf.constant(self._vn_row_splits, tf.int32),
            )

            # VN update -> outgoing VN messages (ragged)
            msg_vn_rg = self._vn_update(msg_vn_rg, llr_vn)

            # EXIT tracking at VNs
            if self._track_exit:
                mi_v = llr2mi(-1.0 * msg_vn_rg.flat_values)
                self._ie_v = tf.tensor_scatter_nd_add(
                    self._ie_v, tf.reshape(it, (1, 1)), tf.reshape(mi_v, (1,))
                )

            # Permute into CN perspective using precomputed ragged mask
            msg_cn_rg = tf.gather(msg_vn_rg.flat_values, self._cn_mask_tf, axis=None)

            # CN update
            msg_cn_rg = self._cn_update(msg_cn_rg)

            # EXIT tracking at CNs
            if self._track_exit:
                mi_c = llr2mi(-1.0 * msg_cn_rg.flat_values)
                self._ie_c = tf.tensor_scatter_nd_add(
                    self._ie_c, tf.reshape(it, (1, 1)), tf.reshape(mi_c, (1,))
                )

            # Back to VN perspective
            msg_out = tf.gather(msg_cn_rg.flat_values, self._ind_cn_inv, axis=None)
            return llr_in, msg_out, it

        def dec_stop(llr_in, msg_in, it):
            return tf.less(it, self._num_iter)

        it0 = tf.constant(0)
        _, msg_vn_flat, _ = tf.while_loop(
            dec_stop,
            dec_iter,
            (llr_flat, msg_vn_flat, it0),
            parallel_iterations=1,
            maximum_iterations=self._num_iter,
        )

        # Optional final message scaling
        if self._trainable_msg_fin:
            msg_vn_flat = tf.ragged.map_flat_values(self._mult_weights, msg_vn_flat, self._edge_weights_fin)

        # Optional final channel-LLR scaling
        if self._trainable_llr_fin:
            llr_fin = tf.ragged.map_flat_values(self._mult_weights, llr_flat, self._llr_weights_fin)
        else:
            llr_fin = llr_flat

        # Final marginalization: LLR + sum of incoming CN messages
        msg_vn_rg = tf.RaggedTensor.from_row_splits(
            values=msg_vn_flat,
            row_splits=tf.constant(self._vn_row_splits, tf.int32),
        )
        x_hat = llr_fin + tf.reduce_sum(msg_vn_rg, axis=1)  # [n, B]
        x_hat = tf.transpose(x_hat, (1, 0))                 # [B, n]

        # Optional hard decision
        if self._hard_out:
            x_hat = tf.cast(tf.less(0.0, x_hat), self._output_dtype)

        # Restore original leading dims
        out_shape = llr_shape
        out_shape[0] = -1
        x_hat = tf.reshape(x_hat, out_shape)

        # Clip and cast output
        x_hat = tf.clip_by_value(x_hat, -self._llr_max_minsum, self._llr_max_minsum)
        x_out = tf.cast(x_hat, self._output_dtype)

        if not self._stateful:
            return x_out
        return x_out, msg_vn_rg
