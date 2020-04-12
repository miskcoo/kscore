#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from kscore.kernels import CurlFreeIMQ
from .base import ScoreEstimator

class SpectralCutoffEstimator(ScoreEstimator):
    def __init__(self,
                 lam=None,
                 keep_rate=None,
                 kernel=CurlFreeIMQ()):
        if lam is not None and keep_rate is not None:
            raise RuntimeError('Cannot specify `lam` and `keep_rate` simultaneously.')
        if lam is None and keep_rate is None:
            raise RuntimeError('Both `lam` and `keep_rate` are `None`.')
        super().__init__(lam, kernel)
        self._keep_rate = keep_rate

    def fit(self, samples, kernel_hyperparams=None):
        # samples: [M, d]
        if kernel_hyperparams is None:
            kernel_hyperparams = self._kernel.heuristic_hyperparams(samples, samples)
        self._kernel_hyperparams = kernel_hyperparams
        self._samples = samples

        M = tf.shape(samples)[-2]
        d = tf.shape(samples)[-1]

        K_op, K_div = self._kernel.kernel_operator(samples, samples,
                kernel_hyperparams=kernel_hyperparams)
        K = K_op.kernel_matrix(flatten=True)

        eigen_values, eigen_vectors = tf.self_adjoint_eig(K)
        if self._keep_rate is not None:
            total_num = tf.shape(K)[0]
            n_eigen = tf.to_int32(tf.to_float(total_num) * self._keep_rate)
        else:
            n_eigen = tf.reduce_sum(tf.to_int32(eigen_values > self._lam))
        # n_eigen = tf.Print(n_eigen, [n_eigen, tf.shape(K)[0]])
        eigen_values = eigen_values[..., -n_eigen:]

        # [Md, eigens], or [M, eigens]
        eigen_vectors = eigen_vectors[..., -n_eigen:] / eigen_values

        H_dh = tf.reduce_sum(K_div, axis=-3)  # [M, d]
        if self._kernel.kernel_type() == 'diagonal':
            truncated_Kinv = tf.expand_dims(eigen_vectors, -2) * eigen_vectors
            truncated_Kinv = tf.reduce_sum(truncated_Kinv, axis=-1) # [M, M]
            self._coeff = tf.matmul(truncated_Kinv, H_dh)
        else:
            H_dh = tf.reshape(H_dh, [-1, 1])   # [Md, 1]
            self._coeff = tf.reduce_sum(eigen_vectors * tf.reduce_sum(
                eigen_vectors * H_dh, axis=0), axis=-1, keep_dims=True)

    def compute_gradients(self, x):
        Kxq_op = self._kernel.kernel_operator(x, self._samples,
                kernel_hyperparams=self._kernel_hyperparams, compute_divergence=False)
        if self._kernel.kernel_type() == 'diagonal':
            Kxq = Kxq_op.kernel_matrix(flatten=True)
            grads = tf.matmul(Kxq, self._coeff)
        else:
            d = tf.shape(x)[-1]
            grads = tf.reshape(Kxq_op.apply(self._coeff), [-1, d])
        return grads
