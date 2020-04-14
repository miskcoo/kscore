#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf

from .base import Base
from kscore.utils import median_heuristic

class SquareCurlFree(Base):

    def __init__(self, kernel_hyperparams=None, heuristic_hyperparams=median_heuristic):
        super().__init__('curl-free', kernel_hyperparams, heuristic_hyperparams)

    def _gram_derivatives(self, x, y, kernel_hyperparams):
        if kernel_hyperparams is None:
            kernel_width = self.heuristic_hyperparams(x, y)
        else:
            kernel_width = kernel_hyperparams
        x_m = tf.expand_dims(x, -2)  # [M, 1, d]
        y_m = tf.expand_dims(y, -3)  # [1, N, d]
        r = x_m - y_m  # [M, N, d]
        norm_rr = tf.reduce_sum(r * r, -1)  # [M, N]
        return self._gram_derivatives_impl(r, norm_rr, kernel_width)

    def _gram_derivatives_impl(self, r, norm_rr, sigma):
        raise NotImplementedError('`_gram_derivatives` not implemented.')

    def kernel_energy(self, x, y, kernel_hyperparams=None, compute_divergence=True):
        d, M, N = tf.shape(x)[-1], tf.shape(x)[-2], tf.shape(y)[-2]
        r, norm_rr, G_1st, G_2nd, _ = self._gram_derivatives(x, y, kernel_hyperparams)

        energy_k = -2. * tf.expand_dims(G_1st, -1) * r

        if compute_divergence:
            divergence = tf.cast(2 * d, G_1st.dtype.base_dtype) * G_1st \
                    + 4. * norm_rr * G_2nd
            return energy_k, divergence
        return energy_k

    def kernel_operator(self, x, y, kernel_hyperparams=None, compute_divergence=True):
        d, M, N = tf.shape(x)[-1], tf.shape(x)[-2], tf.shape(y)[-2]
        r, norm_rr, G_1st, G_2nd, G_3rd = self._gram_derivatives(x, y, kernel_hyperparams)
        G_1st = tf.expand_dims(G_1st, -1)   # [M, N, 1]
        G_2nd = tf.expand_dims(G_2nd, -1)   # [M, N, 1]

        if compute_divergence:
            coeff = (tf.cast(d, G_1st.dtype.base_dtype) + 2) * G_2nd \
                    + 2. * tf.expand_dims(norm_rr * G_3rd, -1)
            divergence = 4. * coeff * r

        def kernel_op(z):
            # z: [N * d, L]
            L = None
            if z.get_shape() is not None:
                L = z.get_shape()[-1]
            if L is None:
                L = tf.shape(z)[-1]
            z = tf.reshape(z, [1, N, d, L])            # [1, N, d, L]
            hat_r = tf.expand_dims(r, -1)              # [M, N, d, 1]
            dot_rz = tf.reduce_sum(z * hat_r, axis=-2) # [M, N,    L]
            coeff = -4. * G_2nd * dot_rz               # [M, N,    L]
            ret = tf.expand_dims(coeff, -2) * hat_r \
                    - 2. * tf.expand_dims(G_1st, -1) * z
            return tf.reshape(tf.reduce_sum(ret, axis=-3), [M * d, L])

        def kernel_adjoint_op(z):
            # z: [M * d, L]
            L = None
            if z.get_shape() is not None:
                L = z.get_shape()[-1]
            if L is None:
                L = tf.shape(z)[-1]
            z = tf.reshape(z, [M, 1, d, L])            # [M, 1, d, L]
            hat_r = tf.expand_dims(r, -1)              # [M, N, d, 1]
            dot_rz = tf.reduce_sum(z * hat_r, axis=-2) # [M, N,    L]
            coeff = -4. * G_2nd * dot_rz               # [M, N,    L]
            ret = tf.expand_dims(coeff, -2) * hat_r \
                    - 2. * tf.expand_dims(G_1st, -1) * z
            return tf.reshape(tf.reduce_sum(ret, axis=-4), [N * d, L])

        def kernel_mat(flatten):
            Km = tf.expand_dims(r, -1) * tf.expand_dims(r, -2)
            K = -2. * tf.expand_dims(G_1st, -1) * tf.eye(d) \
                    - 4. * tf.expand_dims(G_2nd, -1) * Km
            if flatten:
                K = tf.reshape(tf.transpose(K, [0, 2, 1, 3]), [M * d, N * d])
            return K

        linear_operator = collections.namedtuple(
            "KernelOperator", ["shape", "dtype", "apply", "apply_adjoint", "kernel_matrix"])

        op = linear_operator(
            shape=[M * d, N * d],
            dtype=x.dtype,
            apply=kernel_op,
            apply_adjoint=kernel_adjoint_op,
            kernel_matrix=kernel_mat,
        )

        if compute_divergence:
            return op, divergence
        return op
