#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import argparse
import sys
from kscore import *

def generate_data(n_samples):
    theta = tf.random.uniform([n_samples], minval=3.0, maxval=15.0)
    noise = tf.random.normal([n_samples, 2], stddev=np.exp(-1.0))
    samples = tf.linalg.matrix_transpose([
        -2.0 + 2 * theta * tf.cos(theta),
        2 * theta * tf.sin(theta)
    ]) + noise
    return samples

def evaluation_space(size, lower_box, upper_box):
    xs_1d = np.linspace(lower_box, upper_box, size)
    xs = []
    for i in xs_1d:
        for j in xs_1d:
            xs.append([i, j])
    xs = np.array(xs, np.float32)
    return xs

def plot_vector_field(X, Y, normalize=False):
    if normalize:
        for i in range(Y.shape[0]):
            norm = (Y[i][0] ** 2 + Y[i][1] ** 2) ** 0.5
            Y[i] /= norm
    plt.quiver(X[:,0], X[:,1], Y[:,0], Y[:,1])

def clip_energy(energy, threshold=24):
    max_v, min_v = np.max(energy), np.min(energy)
    clip_v = max_v - threshold
    return np.maximum(energy, clip_v) - max_v

def get_estimator(args):
    kernel_dicts = {
        'curlfree_imq': CurlFreeIMQ(),
        'curlfree_rbf': CurlFreeGaussian(),
        'diagonal_imq': DiagonalIMQ(),
        'diagonal_rbf': DiagonalGaussian(),
    }

    estimator_dicts = {
        'tikhonov': TikhonovEstimator,
        'nu': NuEstimator,
        'landweber': LandweberEstimator,
        'spectral_cutoff': SpectralCutoffEstimator,
        'stein': SteinEstimator,
    }

    kernel = kernel_dicts[args.kernel]

    if args.estimator == 'tikhonov_nystrom':
        estimator = TikhonovEstimator(lam=args.lam,
                subsample_rate=args.subsample_rate,
                kernel=kernel_dicts[args.kernel])
    else:
        estimator = estimator_dicts[args.estimator](
                lam=args.lam, kernel=kernel_dicts[args.kernel])
    return estimator

def main(args):
    tf.compat.v1.set_random_seed(1234)
    np.random.seed(1234)

    kernel_width = 8.0
    n_samples = args.n_samples
    size, energy_size = 25, 300
    lower_box, upper_box = -32, 32

    samples = generate_data(n_samples)
    x = evaluation_space(size, lower_box, upper_box)
    x_energy = evaluation_space(energy_size, lower_box, upper_box)

    estimator = get_estimator(args)
    estimator.fit(samples, kernel_hyperparams=kernel_width)

    gradient = estimator.compute_gradients(x)
    if 'curlfree' in args.kernel:
        energy = estimator.compute_energy(x_energy)
    else: energy = tf.constant(0.0)

    with tf.compat.v1.Session() as sess:
        samples, energy, gradient = sess.run([samples, energy, gradient])

    # plot energy
    if 'curlfree' in args.kernel:
        plt.figure(figsize=(4, 4))
        if args.clip_energy:
            energy = clip_energy(energy, threshold=args.clip_threshold)
        img = np.transpose(np.reshape(energy, [energy_size, energy_size]))
        img = np.flip(img, axis=0)
        plt.imshow(img, extent=[lower_box, upper_box, lower_box, upper_box])

    # plot the score field
    plt.figure(figsize=(4, 4))
    plt.scatter(samples[:,0], samples[:,1], 2)
    plot_vector_field(x, gradient)
    plt.show()

if __name__ == "__main__":
    sns.set()
    sns.set_color_codes()
    sns.set_style("white")

    parser = argparse.ArgumentParser()
    parser.add_argument('--estimator', type=str, default='nu', 
            help='score estimator.', choices=['nu', 'landweber', 'tikhonov',
                'tikhonov_nystrom', 'spectral_cutoff', 'stein'])
    parser.add_argument('--n_samples', type=int, default=200, help='sample size.')
    parser.add_argument('--lam', type=float, default=1.0e-5, help='regularization parameter.')
    parser.add_argument('--kernel', type=str, default='curlfree_imq',
            help='matrix-valued kernel.', choices=['curlfree_imq',
                'curlfree_rbf', 'diagonal_imq', 'diagonal_rbf'])
    parser.add_argument('--subsample_rate', default=None, type=float,
            help='subsample rate used in the Nystrom approimation.')
    parser.add_argument('--clip_energy', default=True, type=bool,
            help='whether to clip the energy function.')
    parser.add_argument('--clip_threshold', default=24, type=int)
    args = parser.parse_args(sys.argv[1:])

    main(args)
