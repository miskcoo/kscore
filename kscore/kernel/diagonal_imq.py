#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf

from .diagonal import DiagonalKernel
from kscore.utils import median_heuristic

class DiagonalIMQ(DiagonalKernel):

    def __init__(self, kernel_hyperparams=None, heuristic_hyperparams=median_heuristic):
        super().__init__(kernel_hyperparams, heuristic_hyperparams)

    def _gram(self, x, y, kernel_hyperparams):
        if kernel_hyperparams is None:
            kernel_width = self.heuristic_hyperparams(x, y)
        else:
            kernel_width = kernel_hyperparams

        d = tf.shape(x)[-1]
        x_m = tf.expand_dims(x, -2)  # [M, 1, d]
        y_m = tf.expand_dims(y, -3)  # [1, N, d]
        diff = x_m - y_m
        dist2 = tf.reduce_sum(diff * diff, -1) # [M, N]
        imq = tf.rsqrt(1 + dist2 / kernel_width ** 2)
        divergence = tf.expand_dims(imq ** 3, -1) * (diff / kernel_width ** 2)

        return imq, divergence
