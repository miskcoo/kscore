#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from kscore.kernels import DiagonalIMQ
from .base import Base

class Stein(Base):
    def __init__(self,
                 lam,
                 kernel=DiagonalIMQ(),
                 dtype=tf.float32):
        # TODO: Implement curl-free kernels
        if kernel.kernel_type() != 'diagonal':
            raise NotImplementedError('Only support diagonal kernels.')
        super().__init__(lam, kernel, dtype)

    def fit(self, samples, kernel_hyperparams=None):
        if kernel_hyperparams is None:
            kernel_hyperparams = self._kernel.heuristic_hyperparams(samples, samples)
        self._kernel_hyperparams = kernel_hyperparams
        self._samples = samples

        M = tf.shape(samples)[-2]
        K_op, K_div = self._kernel.kernel_operator(samples, samples,
                kernel_hyperparams=kernel_hyperparams)
        K = K_op.kernel_matrix(flatten=True)
        # In the Stein estimator (Li & Turner, 2018), the regularization parameter is divided
        # by $M^2$, for the unified meaning of $\lambda$, we multiply this back.
        Mlam = tf.cast(M, self._dtype) ** 2 * self._lam
        Kinv = tf.linalg.inv(K + Mlam * tf.eye(M))
        H_dh = tf.reduce_sum(K_div, axis=-2)
        grads = -tf.matmul(Kinv, H_dh)
        self._coeff = { 'Kinv': Kinv, 'grads': grads, 'Mlam': Mlam }

    def _compute_gradients_one(self, x):
        # Section 3.4 in Li & Turner (2018), the our-of-sample extension.
        Kxx = self._kernel.kernel_matrix(x, x,
                kernel_hyperparams=self._kernel_hyperparams, compute_divergence=False)
        Kqx, Kqx_div = self._kernel.kernel_matrix(self._samples, x,
                kernel_hyperparams=self._kernel_hyperparams)
        KxqKinv = tf.matmul(Kqx, self._coeff['Kinv'], transpose_a=True)
        term1 = -1. / (Kxx + self._coeff['Mlam'] - tf.matmul(KxqKinv, Kqx))
        term2 = tf.matmul(Kqx, self._coeff['grads'], transpose_a=True) \
                - tf.matmul(KxqKinv + 1., tf.squeeze(Kqx_div, -2))
        return tf.matmul(term1, term2)

    def compute_gradients(self, x):
        if x is self._samples:
            return self._coeff['grads']
        else:
            def stein_dlog(y):
                stein_dlog_qx = self._compute_gradients_one(tf.expand_dims(y, 0))
                stein_dlog_qx = tf.squeeze(stein_dlog_qx, axis=-2)
                return stein_dlog_qx

            return tf.map_fn(stein_dlog, x)
