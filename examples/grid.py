#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import argparse
import kscore

from .utils import *

def generate_distribution(dim):
    grid_vertices = np.random.randint(low=0, high=2, size=[dim, dim])
    grid_vertices = tf.cast(grid_vertices, tf.float32) # * 2.0 - 1.0

    def sampler(n_samples):
        q_id = tf.random.categorical(tf.log(tf.cast(dim, tf.float32)) * tf.zeros([1, dim]), n_samples)
        q = tf.random.normal([n_samples, dim, dim]) + grid_vertices
        samples_many = tf.transpose(q, [0, 1, 2])
        samples_id = tf.transpose(tf.concat([[tf.range(0, n_samples, dtype=tf.int64)], q_id], 0))
        return tf.gather_nd(samples_many, samples_id)

    def log_prob(x):
        m = tf.expand_dims(x, -2) - grid_vertices  # [M, D, D]
        prob = tf.reduce_logsumexp(-0.5 * tf.reduce_sum(tf.square(m), -1), axis=-1)
        return prob - 0.5 * tf.log(3.1415926535897) - tf.log(tf.cast(dim, tf.float32))

    def grad_log_prob(x):
        return tf.map_fn(lambda i: tf.gradients(log_prob(i), i)[0], x)

    return sampler, grad_log_prob

def compute_l2_dist(x, y):
    l2_d = tf.reduce_sum(tf.square(x - y), axis=-1)
    return tf.reduce_mean(l2_d)

def main(args):
    sampler, grad_log_prob = generate_distribution(args.dim)
    x_train, x_test = sampler(args.n_data), sampler(args.n_test)
    estimator = get_estimator(args)
    estimator.fit(x_train)
    y_test = estimator.compute_gradients(x_test)
    l2_d = compute_l2_dist(y_test, grad_log_prob(x_test))

    with tf.Session() as sess:
        l2_d = sess.run(l2_d)

    print("L2 distance = %.4lf" % l2_d)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_estimator_params(parser)
    parser.add_argument('--n_data', type=int, default=128, help='data sample size.')
    parser.add_argument('--n_test', type=int, default=1024, help='test sample size.')
    parser.add_argument('--dim', type=int, default=128, help='dimension.')
    args = parser.parse_args(sys.argv[1:])
    main(args)
