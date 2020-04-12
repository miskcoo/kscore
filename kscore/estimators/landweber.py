#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.solvers as solvers

from kscore.kernels import CurlFreeIMQ
from .base import ScoreEstimator

class LandweberEstimator(ScoreEstimator):
    def __init__(self,
                 lam=None,
                 iternum=None,
                 kernel=CurlFreeIMQ(),
                 stepsize=1.0e-2):
        if lam is not None and iternum is not None:
            raise RuntimeError('Cannot specify `lam` and `iternum` simultaneously.')
        if lam is None and iternum is None:
            raise RuntimeError('Both `lam` and `iternum` are `None`.')
        if iternum is not None:
            iternum = int(iternum + 0.5)
            lam = 1.0 / iternum
        else:
            iternum = int(1.0 / lam) + 1
        super().__init__(lam, kernel)
        self._stepsize = tf.to_float(stepsize)
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
                    - (tf.to_float(t)) * self._stepsize ** 2 * H_dh
            return (t + 1, nc)

        t, coeff = tf.while_loop(
            lambda t, c: t < tf.to_int32(self._iternum),
            get_next,
            loop_vars=[1, tf.zeros_like(H_dh)]
        )

        self._coeff = (-tf.to_float(t) * self._stepsize, coeff)

    def compute_gradients(self, x):
        d = tf.shape(x)[-1]
        Kxq_op, div_xq = self._kernel.kernel_operator(x, self._samples,
                kernel_hyperparams=self._kernel_hyperparams)
        div_xq = tf.reduce_mean(div_xq, axis=-2) * self._coeff[0]
        grads = Kxq_op.apply(self._coeff[1])
        grads = tf.reshape(grads, [-1, d]) + div_xq
        return grads
