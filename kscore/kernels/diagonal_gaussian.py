#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf

from .diagonal import Diagonal
from kscore.utils import median_heuristic

class DiagonalGaussian(Diagonal):

    def __init__(self, kernel_hyperparams=None, heuristic_hyperparams=median_heuristic):
        super().__init__(kernel_hyperparams, heuristic_hyperparams)

    def _gram_impl(self, x, y, kernel_width):
        d = tf.shape(x)[-1]
        x_m = tf.expand_dims(x, -2)  # [M, 1, d]
        y_m = tf.expand_dims(y, -3)  # [1, N, d]
        diff = x_m - y_m
        dist2 = tf.reduce_sum(diff * diff, -1) # [M, N]
        rbf = tf.exp(-0.5 * dist2 / kernel_width ** 2) # [M, N]
        divergence = tf.expand_dims(rbf, -1) * (diff / kernel_width ** 2)

        return rbf, divergence
