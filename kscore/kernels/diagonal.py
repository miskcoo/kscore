#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf

from .base import Base
from kscore.utils import median_heuristic

class Diagonal(Base):

    def __init__(self, kernel_hyperparams, heuristic_hyperparams):
        super().__init__('diagonal', kernel_hyperparams, heuristic_hyperparams)

    def _gram(self, x, y, kernel_hyperparams):
        if kernel_hyperparams is None:
            kernel_width = self.heuristic_hyperparams(x, y)
        else:
            kernel_width = kernel_hyperparams
        return self._gram_impl(x, y, kernel_width)

    def _gram_impl(self, x, y, kernel_hyperparams):
        raise NotImplementedError('Gram matrix not implemented!')

    def kernel_operator(self, x, y, kernel_hyperparams=None, compute_divergence=True):
        d = tf.shape(x)[-1]
        M = tf.shape(x)[-2]
        N = tf.shape(y)[-2]
        K, divergence = self._gram(x, y, kernel_hyperparams)

        def kernel_op(z):
            # z: [N * d, L]
            L = None
            if z.get_shape() is not None:
                L = z.get_shape()[-1]
            if L is None:
                L = tf.shape(z)[-1]
            z = tf.reshape(z, [N, d * L])
            ret = tf.matmul(K, z)
            return tf.reshape(ret, [M * d, L])

        def kernel_adjoint_op(z):
            # z: [M * d, L]
            L = None
            if z.get_shape() is not None:
                L = z.get_shape()[-1]
            if L is None:
                L = tf.shape(z)[-1]
            z = tf.reshape(z, [M, d * L])
            ret = tf.matmul(K, z, transpose_a=True)
            return tf.reshape(ret, [N * d, L])

        def kernel_mat(flatten):
            if flatten:
                return K
            return tf.expand_dims(tf.expand_dims(K, -1), -1) * tf.eye(d)

        linear_operator = collections.namedtuple(
            "Operator", ["shape", "dtype", "apply", "apply_adjoint", "kernel_matrix"])

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

