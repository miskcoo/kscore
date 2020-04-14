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
import kscore

from .utils import *

def generate_data(n_samples):
    theta = tf.random.uniform([n_samples], minval=3.0, maxval=15.0)
    noise = tf.random.normal([n_samples, 2], stddev=np.exp(-1.0))
    samples = tf.linalg.matrix_transpose([
        -2.0 + 2 * theta * tf.cos(theta),
        2 * theta * tf.sin(theta)
    ]) + noise
    return samples

def clip_energy(energy, threshold=24):
    max_v, min_v = np.max(energy), np.min(energy)
    clip_v = max_v - threshold
    return np.maximum(energy, clip_v) - max_v

def main(args):
    tf.compat.v1.set_random_seed(1234)
    np.random.seed(1234)

    kernel_width = 8.0
    n_samples = args.n_samples
    size, energy_size = 25, 300
    lower_box, upper_box = -args.plot_range, args.plot_range

    samples = generate_data(n_samples)
    x = linspace_2d(size, lower_box, upper_box)
    x_energy = linspace_2d(energy_size, lower_box, upper_box)

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
    add_estimator_params(parser)
    parser.add_argument('--n_samples', type=int, default=200, help='sample size.')
    parser.add_argument('--plot_range', default=32, type=int)
    parser.add_argument('--clip_energy', default=True, type=bool,
            help='whether to clip the energy function.')
    parser.add_argument('--clip_threshold', default=24, type=int)
    args = parser.parse_args(sys.argv[1:])

    main(args)
