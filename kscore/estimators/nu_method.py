#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math

from kscore.kernels import CurlFreeIMQ
from .base import Base

class NuMethod(Base):
    def __init__(self,
                 lam=None,
                 iternum=None,
                 kernel=CurlFreeIMQ(),
                 nu=1.0,
                 dtype=tf.float32):
        if lam is not None and iternum is not None:
            raise RuntimeError('Cannot specify `lam` and `iternum` simultaneously.')
        if lam is None and iternum is None:
            raise RuntimeError('Both `lam` and `iternum` are `None`.')
        if iternum is not None:
            lam = 1.0 / tf.cast(iternum, dtype) ** 2
        else:
            iternum = tf.cast(1.0 / tf.sqrt(lam), tf.int32) + 1
        super().__init__(lam, kernel, dtype)
        self._nu = nu
        self._iternum = iternum

    def fit(self, samples, kernel_hyperparams=None):
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

        def get_next(t, a, pa, c, pc):
            # nc <- c <- pc
            ft = tf.cast(t, self._dtype)
            nu = self._nu
            u = (ft - 1.) * (2. * ft - 3.) * (2. * ft + 2. * nu - 1.) \
                    / ((ft + 2. * nu - 1.) * (2. * ft + 4. * nu - 1.) * (2. * ft + 2. * nu - 3.))
            w = 4. * (2. * ft + 2. * nu - 1.) * (ft + nu - 1.) / ((ft + 2. * nu - 1.) * (2. * ft + 4. * nu - 1.))
            nc = (1. + u) * c - w * (a * H_dh + K_op.apply(c)) / tf.cast(M, self._dtype) - u * pc
            na = (1. + u) * a - u * pa - w
            return (t + 1, na, a, nc, c)

        a1 = -(4. * self._nu + 2) / (4. * self._nu + 1)
        ret = tf.while_loop(
            lambda t, a, pa, c, pc: t <= tf.cast(self._iternum, tf.int32),
            get_next,
            loop_vars=[2, a1, 0., tf.zeros_like(H_dh), tf.zeros_like(H_dh)]
        )

        self._coeff = (ret[1], ret[3])

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
