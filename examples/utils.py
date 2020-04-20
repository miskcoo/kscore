#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import kscore
from skimage import io, img_as_ubyte
from skimage.exposure import rescale_intensity

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

def makedirs(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

def setup_logger(name, src, result_path, filename="log"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    log_path = os.path.join(result_path, filename)
    makedirs(log_path)
    info_file_handler = logging.FileHandler(log_path)
    info_file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(info_file_handler)
    logger.addHandler(console_handler)
    logger.info(src)
    with open(src) as f:
        logger.info(f.read())
    return logger

def save_image_collections(x, filename, shape=(10, 10), scale_each=False, transpose=False):
    makedirs(filename)
    n = x.shape[0]
    if transpose:
        x = x.transpose(0, 2, 3, 1)
    if scale_each is True:
        for i in range(n):
            x[i] = rescale_intensity(x[i], out_range=(0, 1))
    n_channels = x.shape[3]
    x = img_as_ubyte(x)
    r, c = shape
    if r * c < n:
        print('Shape too small to contain all images')
    h, w = x.shape[1:3]
    ret = np.zeros((h * r, w * c, n_channels), dtype='uint8')
    for i in range(r):
        for j in range(c):
            if i * c + j < n:
                ret[i * h:(i + 1) * h, j * w:(j + 1) * w, :] = x[i * c + j]
    ret = ret.squeeze()
    io.imsave(filename, ret)
    return ret
