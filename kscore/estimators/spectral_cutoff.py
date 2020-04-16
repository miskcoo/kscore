#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from kscore.kernels import CurlFreeIMQ
from .base import Base

class SpectralCutoff(Base):
    def __init__(self,
                 lam=None,
                 keep_rate=None,
                 kernel=CurlFreeIMQ(),
                 dtype=tf.float32):
        if lam is not None and keep_rate is not None:
            raise RuntimeError('Cannot specify `lam` and `keep_rate` simultaneously.')
        if lam is None and keep_rate is None:
            raise RuntimeError('Both `lam` and `keep_rate` are `None`.')
        super().__init__(lam, kernel, dtype)
        self._keep_rate = keep_rate

    def fit(self, samples, kernel_hyperparams=None):
        if kernel_hyperparams is None:
            kernel_hyperparams = self._kernel.heuristic_hyperparams(samples, samples)
        self._kernel_hyperparams = kernel_hyperparams
        self._samples = samples

        M = tf.shape(samples)[-2]
        d = tf.shape(samples)[-1]

        K_op, K_div = self._kernel.kernel_operator(samples, samples,
                kernel_hyperparams=kernel_hyperparams)
        K = K_op.kernel_matrix(flatten=True)

        eigen_values, eigen_vectors = tf.linalg.eigh(K / tf.cast(M, self._dtype))
        if self._keep_rate is not None:
            total_num = tf.shape(K)[0]
            n_eigen = tf.cast(tf.cast(total_num, self._dtype) * self._keep_rate, tf.int32)
        else:
            n_eigen = tf.reduce_sum(tf.cast(eigen_values > self._lam, tf.int32))
        # eigen_values = tf.Print(eigen_values, [eigen_values[0], eigen_values[-1]])
        n_eigen = tf.maximum(n_eigen, 1)
        eigen_values = eigen_values[..., -n_eigen:]

        # [Md, eigens], or [M, eigens]
        eigen_vectors = eigen_vectors[..., -n_eigen:] / eigen_values

        H_dh = tf.reduce_mean(K_div, axis=-2)  # [M, d]
        if self._kernel.kernel_type() == 'diagonal':
            truncated_Kinv = tf.expand_dims(eigen_vectors, -2) * eigen_vectors
            truncated_Kinv = tf.reduce_sum(truncated_Kinv, axis=-1) # [M, M]
            self._coeff = tf.matmul(truncated_Kinv, H_dh)
        else:
            H_dh = tf.reshape(H_dh, [-1, 1])   # [Md, 1]
            self._coeff = tf.reduce_sum(eigen_vectors * tf.reduce_sum(
                eigen_vectors * H_dh, axis=0), axis=-1, keepdims=True)
        self._coeff /= tf.cast(M, self._dtype)

    def _compute_energy(self, x):
        Kxq = self._kernel.kernel_energy(x, self._samples,
                kernel_hyperparams=self._kernel_hyperparams,
                compute_divergence=False)
        Kxq = tf.reshape(Kxq, [tf.shape(x)[-2], -1])
        energy = tf.matmul(Kxq, self._coeff)
        return -tf.reshape(energy, [-1])

    def compute_gradients(self, x):
        Kxq_op = self._kernel.kernel_operator(x, self._samples,
                kernel_hyperparams=self._kernel_hyperparams, compute_divergence=False)
        if self._kernel.kernel_type() == 'diagonal':
            Kxq = Kxq_op.kernel_matrix(flatten=True)
            grads = tf.matmul(Kxq, self._coeff)
        else:
            d = tf.shape(x)[-1]
            grads = tf.reshape(Kxq_op.apply(self._coeff), [-1, d])
        return -grads
