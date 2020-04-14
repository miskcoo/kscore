#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from kscore import *

def generate_data(n_samples):
    theta = tf.random.uniform([n_samples], minval=3.0, maxval=15.0)
    noise = tf.random.normal([n_samples, 2], stddev=np.exp(-1.0))
    samples = tf.linalg.matrix_transpose([
        -2.0 + 2 * theta * tf.cos(theta),
        2 * theta * tf.sin(theta)
    ]) + noise
    return samples

def evaluation_space(size, energy_size, lower_box, upper_box):
    xs_1d = np.linspace(lower_box, upper_box, size)
    xs = []
    for i in xs_1d:
        for j in xs_1d:
            xs.append([i, j])
    xs = np.array(xs, np.float32)
    x = xs

    xs_1d = np.linspace(lower_box, upper_box, energy_size)
    xs_energy = []
    for i in xs_1d:
        for j in xs_1d:
            xs_energy.append([i, j])
    xs_energy = np.array(xs_energy, np.float32)
    x_energy = xs_energy
    return x, x_energy

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

def main():
    tf.compat.v1.set_random_seed(1234)
    np.random.seed(1234)

    kernel_width = 8.0
    n_samples = 200
    size, energy_size = 25, 300
    lower_box, upper_box = -32, 32

    samples = generate_data(n_samples)
    x, x_energy = evaluation_space(size, energy_size, lower_box, upper_box)

    estimator = NuEstimator(lam=0.00001, kernel=CurlFreeIMQp(0.5))
    estimator.fit(samples, kernel_hyperparams=kernel_width)

    energy = estimator.compute_energy(x_energy)
    gradient = estimator.compute_gradients(x)

    with tf.compat.v1.Session() as sess:
        samples, energy, gradient = sess.run([samples, energy, gradient])

    # plot energy
    plt.figure(figsize=(4, 4))
    energy = clip_energy(energy)
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
    main()
