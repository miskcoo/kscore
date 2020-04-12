#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class ScoreEstimator:
    def __init__(self, lam, kernel):
        self._lam = lam
        self._kernel = kernel
        self._coeff = None
        self._kernel_hyperparams = None
        self._samples = None

    def fit(self, samples, kernel_hyperparams):
        raise NotImplementedError('Not implemented score estimator!')

    def compute_gradients(self, x):
        raise NotImplementedError('Not implemented score estimator!')

    def compute_energy(self, x):
        raise NotImplementedError('Not implemented score estimator!')
