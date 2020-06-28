#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf

from .square_curlfree import SquareCurlFree
from kscore.utils import median_heuristic

class CurlFreeGaussian(SquareCurlFree):
    def __init__(self, kernel_hyperparams=None, heuristic_hyperparams=median_heuristic):
        super().__init__(kernel_hyperparams, heuristic_hyperparams)

    def _gram_derivatives_impl(self, r, norm_rr, sigma):
        ''' 
        Construct the curl-free kernel $-\nabla^2 \psi(\|x - y\|^2)$.
        You need to provide the first, second and third derivatives of $\psi$.
        See eq. (21) and eq. (22). 
        '''
        inv_sqr_sigma = 0.5 / tf.square(sigma)
        rbf = tf.exp(-norm_rr * inv_sqr_sigma)
        G_1st = -rbf * inv_sqr_sigma
        G_2nd = -G_1st * inv_sqr_sigma
        G_3rd = -G_2nd * inv_sqr_sigma
        return r, norm_rr, G_1st, G_2nd, G_3rd
