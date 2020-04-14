#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class Base:
    def __init__(self, lam, kernel, dtype):
        self._lam = lam
        self._kernel = kernel
        self._coeff = None
        self._kernel_hyperparams = None
        self._samples = None
        self._dtype = dtype

    def fit(self, samples, kernel_hyperparams):
        raise NotImplementedError('Not implemented score estimator!')

    def compute_gradients(self, x):
        raise NotImplementedError('Not implemented score estimator!')

    def compute_energy(self, x):
        if self._kernel.kernel_type() != 'curl-free':
            raise RuntimeError('Only curl-free kernels have well-defined energy.')
        return self._compute_energy(x)

    def _compute_energy(self, x):
        raise NotImplementedError('Not implemented score estimator!')
