#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf

from .square_curlfree import SquareCurlFree
from kscore.utils import median_heuristic

class CurlFreeIMQ(SquareCurlFree):
    def __init__(self, kernel_hyperparams=None, heuristic_hyperparams=median_heuristic):
        super().__init__(kernel_hyperparams, heuristic_hyperparams)

    def _gram_derivatives_impl(self, r, norm_rr, sigma):
        inv_sqr_sigma = 1.0 / tf.square(sigma)
        imq = tf.math.rsqrt(1.0 + norm_rr * inv_sqr_sigma) # [M, N]
        imq_2 = 1.0 / (1.0 + norm_rr * inv_sqr_sigma)
        G_1st = -0.5 * imq_2 * inv_sqr_sigma * imq
        G_2nd = -1.5 * imq_2 * inv_sqr_sigma * G_1st
        G_3rd = -2.5 * imq_2 * inv_sqr_sigma * G_2nd
        return r, norm_rr, G_1st, G_2nd, G_3rd

class CurlFreeIMQp(SquareCurlFree):
    def __init__(self, p=0.5, kernel_hyperparams=None, heuristic_hyperparams=median_heuristic):
        super().__init__(kernel_hyperparams, heuristic_hyperparams)
        self._p = p

    def _gram_derivatives_impl(self, r, norm_rr, sigma):
        inv_sqr_sigma = 1.0 / tf.square(sigma)
        imq = 1.0 / (1.0 + norm_rr * inv_sqr_sigma)
        imq_p = tf.pow(imq, self._p) # [M, N]
        G_1st = -(0. + self._p) * imq * inv_sqr_sigma * imq_p
        G_2nd = -(1. + self._p) * imq * inv_sqr_sigma * G_1st
        G_3rd = -(2. + self._p) * imq * inv_sqr_sigma * G_2nd
        return r, norm_rr, G_1st, G_2nd, G_3rd
