#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import kscore

def get_estimator(args):
    kernel_dicts = {
        'curlfree_imq': kscore.kernels.CurlFreeIMQ,
        'curlfree_rbf': kscore.kernels.CurlFreeGaussian,
        'diagonal_imq': kscore.kernels.DiagonalIMQ,
        'diagonal_rbf': kscore.kernels.DiagonalGaussian,
    }

    estimator_dicts = {
        'tikhonov': kscore.estimators.Tikhonov,
        'nu': kscore.estimators.NuMethod,
        'landweber': kscore.estimators.Landweber,
        'spectral_cutoff': kscore.estimators.SpectralCutoff,
        'stein': kscore.estimators.Stein,
    }

    if args.estimator == 'tikhonov_nystrom':
        estimator = kscore.estimators.Tikhonov(lam=args.lam,
                subsample_rate=args.subsample_rate,
                kernel=kernel_dicts[args.kernel]())
    else:
        estimator = estimator_dicts[args.estimator](
                lam=args.lam, kernel=kernel_dicts[args.kernel]())
    return estimator

def add_estimator_params(parser):
    parser.add_argument('--estimator', type=str, default='nu', 
            help='score estimator.', choices=['nu', 'landweber', 'tikhonov',
                'tikhonov_nystrom', 'spectral_cutoff', 'stein'])
    parser.add_argument('--lam', type=float, default=1.0e-5, help='regularization parameter.')
    parser.add_argument('--kernel', type=str, default='curlfree_imq',
            help='matrix-valued kernel.', choices=['curlfree_imq',
                'curlfree_rbf', 'diagonal_imq', 'diagonal_rbf'])
    parser.add_argument('--subsample_rate', default=None, type=float,
            help='subsample rate used in the Nystrom approimation.')
    return parser

def plot_vector_field(X, Y, normalize=False):
    if normalize:
        for i in range(Y.shape[0]):
            norm = (Y[i][0] ** 2 + Y[i][1] ** 2) ** 0.5
            Y[i] /= norm
    plt.quiver(X[:,0], X[:,1], Y[:,0], Y[:,1])

def linspace_2d(size, lower_box, upper_box):
    xs_1d = np.linspace(lower_box, upper_box, size)
    xs = []
    for i in xs_1d:
        for j in xs_1d:
            xs.append([i, j])
    xs = np.array(xs, np.float32)
    return xs

def clip_energy(energy, threshold=24):
    max_v, min_v = np.max(energy), np.min(energy)
    clip_v = max_v - threshold
    return np.maximum(energy, clip_v) - max_v
