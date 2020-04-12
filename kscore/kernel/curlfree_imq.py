#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf

from .base import BaseKernel
from kscore.utils import median_heuristic

class CurlFreeIMQ(BaseKernel):

    def __init__(self, kernel_width=None, heuristic_hyperparams=median_heuristic):
        if kernel_width is not None:
            heuristic_hyperparams = lambda x, y: kernel_width
        super().__init__('curl-free', heuristic_hyperparams)

    def kernel_operator(self, x, y, kernel_hyperparams=None, compute_divergence=True):
        if kernel_hyperparams is None:
            kernel_width = self.heuristic_hyperparams(x, y)
        else:
            kernel_width = kernel_hyperparams

        x_m = tf.expand_dims(x, -2)  # [M, 1, d]
        y_m = tf.expand_dims(y, -3)  # [1, N, d]
        d = tf.shape(x)[-1]
        M = tf.shape(x)[-2]
        N = tf.shape(y)[-2]
        diff = x_m - y_m  # [M, N, d]
        dist2 = tf.reduce_sum(diff * diff, -1)  # [M, N]
        inv_sqr_sigma = 1.0 / tf.square(kernel_width)
        imq = tf.rsqrt(1.0 + dist2 * inv_sqr_sigma) # [M, N]
        imq_2 = 1 / (1.0 + dist2 * inv_sqr_sigma)
        imq_3 = imq * imq_2

        if compute_divergence:
            imq_5 = imq_3 * imq_2
            div_coeff = 3 * imq_5 * (5 * dist2 * inv_sqr_sigma * imq_2 - (tf.to_float(d) + 2)) * inv_sqr_sigma ** 2 # [M, N]
            divergence = -tf.expand_dims(div_coeff, -1) * diff 

        def kernel_op(z):
            # z: [N * d, L]
            L = None
            if z.get_shape() is not None:
                L = z.get_shape()[-1]
            if L is None:
                L = tf.shape(z)[-1]
            # print(N, d, L)
            z = tf.reshape(z, [1, N, d, L]) # [1, N, d, L]
            r = tf.expand_dims(diff, -1) / kernel_width  # [M, N, d, 1]
            dot_rz = tf.reduce_sum(z * r, axis=-2) # [M, N, L]
            coeff = tf.expand_dims(tf.expand_dims(imq_3 * inv_sqr_sigma, -1), -1) # [M, N, 1, 1]
            coeff_i = tf.expand_dims(tf.expand_dims(3 * imq_2, -1), -1) # [M, N, 1, 1]
            ret = coeff * (z - coeff_i * tf.expand_dims(dot_rz, -2) * r) # [M, N, d, L]
            return tf.reshape(tf.reduce_sum(ret, axis=-3), [M * d, L])

        def kernel_transpose_op(z):
            # z: [M * d, L]
            L = None
            if z.get_shape() is not None:
                L = z.get_shape()[-1]
            if L is None:
                L = tf.shape(z)[-1]
            z = tf.reshape(z, [M, 1, d, L]) # [M, 1, d, L]
            r = tf.expand_dims(diff, -1) / kernel_width  # [M, N, d, 1]
            dot_rz = tf.reduce_sum(z * r, axis=-2) # [M, N, L]
            coeff = tf.expand_dims(tf.expand_dims(imq_3 * inv_sqr_sigma, -1), -1) # [M, N, 1, 1]
            coeff_i = tf.expand_dims(tf.expand_dims(3 * imq_2, -1), -1) # [M, N, 1, 1]
            ret = coeff * (z - coeff_i * tf.expand_dims(dot_rz, -2) * r) # [M, N, d, L]
            return tf.reshape(tf.reduce_sum(ret, axis=-4), [N * d, L])

        def kernel_mat(flatten):
            Km = tf.expand_dims(diff, -1) * tf.expand_dims(diff, -2) * inv_sqr_sigma
            coeff = tf.expand_dims(tf.expand_dims(imq_3 * inv_sqr_sigma, -1), -1) # [M, N, 1, 1]
            coeff_i = tf.expand_dims(tf.expand_dims(3 * imq_2, -1), -1) # [M, N, 1, 1]
            K = coeff * (tf.eye(d) - coeff_i * Km) # [M, N, d, d]
            if flatten:
                K = tf.reshape(tf.transpose(K, [0, 2, 1, 3]), [M * d, N * d])
            return K

        linear_operator = collections.namedtuple(
            "KernelOperator", ["shape", "dtype", "apply", "apply_transpose", "kernel_matrix"])

        op = linear_operator(
            shape=[M * d, N * d],
            dtype=x.dtype,
            apply=kernel_op,
            apply_transpose=kernel_transpose_op,
            kernel_matrix=kernel_mat,
        )

        if compute_divergence:
            return op, divergence
        return op

    def kernel_matrix(self, x, y, kernel_hyperparams=None, flatten=False, compute_divergence=True):
        if compute_divergence:
            op, divergence = self.kernel_operator(x, y, True, kernel_width)
            return op.kernel_matrix(flatten), divergence
        op = self.kernel_operator(x, y, False, kernel_width)
        return op.kernel_matrix(flatten)
