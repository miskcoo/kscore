#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from kscore.kernels import CurlFreeIMQ
from .base import Base

class Landweber(Base):
    def __init__(self,
                 lam=None,
                 iternum=None,
                 kernel=CurlFreeIMQ(),
                 stepsize=1.0e-2,
                 dtype=tf.float32):
        if lam is not None and iternum is not None:
            raise RuntimeError('Cannot specify `lam` and `iternum` simultaneously.')
        if lam is None and iternum is None:
            raise RuntimeError('Both `lam` and `iternum` are `None`.')
        if iternum is not None:
            lam = 1.0 / tf.cast(iternum, dtype)
        else:
            iternum = tf.cast(1.0 / lam, tf.int32) + 1
        super().__init__(lam, kernel, dtype)
        self._stepsize = tf.cast(stepsize, dtype)
        self._iternum = iternum

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

        # H_dh: [Md, 1]
        H_dh = tf.reshape(tf.reduce_mean(K_div, axis=-2), [M * d, 1])

        def get_next(t, c):
            nc = c - self._stepsize * K_op.apply(c) \
                    - (tf.cast(t, self._dtype)) * self._stepsize ** 2 * H_dh
            return (t + 1, nc)

        t, coeff = tf.while_loop(
            lambda t, c: t < tf.cast(self._iternum, tf.int32),
            get_next,
            loop_vars=[1, tf.zeros_like(H_dh)]
        )

        self._coeff = (-tf.cast(t, self._dtype) * self._stepsize, coeff)

    def _compute_energy(self, x):
        Kxq, div_xq = self._kernel.kernel_energy(x, self._samples,
                kernel_hyperparams=self._kernel_hyperparams)
        Kxq = tf.reshape(Kxq, [tf.shape(x)[-2], -1])
        div_xq = tf.reduce_mean(div_xq, axis=-1) * self._coeff[0]
        energy = tf.reshape(tf.matmul(Kxq, self._coeff[1]), [-1]) + div_xq
        return energy

    def compute_gradients(self, x):
        d = tf.shape(x)[-1]
        Kxq_op, div_xq = self._kernel.kernel_operator(x, self._samples,
                kernel_hyperparams=self._kernel_hyperparams)
        div_xq = tf.reduce_mean(div_xq, axis=-2) * self._coeff[0]
        grads = Kxq_op.apply(self._coeff[1])
        grads = tf.reshape(grads, [-1, d]) + div_xq
        return grads
