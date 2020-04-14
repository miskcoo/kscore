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
import kscore.utils

from .utils import *

def compute_log_prob(x, q_mean):
    m = tf.expand_dims(x, -2) - q_mean  # [M, D, D]
    prob = tf.reduce_logsumexp(-0.5 * tf.reduce_sum(tf.square(m), -1), axis=-1)
    return prob - 0.5 * tf.log(3.1415926535897) - np.log(float(D))

def compute_l2_dist(x, y):
    l2_d = tf.reduce_sum(tf.square(x - y), axis=-1)
    return tf.reduce_mean(l2_d)

def get_estimator(args, lam, sigma):
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

    kernel = kernel_dicts[args.kernel](kernel_hyperparams=sigma)

    if args.estimator == 'tikhonov_nystrom':
        estimator = kscore.estimators.Tikhonov(lam=lam,
                subsample_rate=args.subsample_rate, kernel=kernel)
    if args.estimator == 'tikhonov':
        estimator = kscore.estimators.Tikhonov(lam=lam, use_cg=True, kernel=kernel)
    else:
        estimator = estimator_dicts[args.estimator](lam=lam, kernel=kernel)
    return estimator

def main(args, sess, repeat=4):
    dim = args.dim
    losses = []
    lam = tf.placeholder(tf.float32, shape=[])
    sigma = tf.placeholder(tf.float32, shape=[])

    for _ in range(repeat):
        grid_vertices = np.random.randint(low=0, high=2, size=[dim, dim]).astype(np.float32)

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

        x_train, x_test = sampler(args.n_data), sampler(args.n_test)
        x_train, x_test = sess.run([x_train, x_test])

        estimator = get_estimator(args, lam, sigma)
        estimator.fit(x_train)
        l2_loss = compute_l2_dist(estimator.compute_gradients(x_test), grad_log_prob(x_test))
        losses.append(l2_loss)

    l2_loss = tf.reduce_mean(losses)

    sigma_middle = (3.0 * args.dim) ** 0.5
    sigma_space = np.linspace(sigma_middle * 0.25, sigma_middle * 4)
    lam_space = np.logspace(-8.5, 2)

    results = []
    f = open('%s-%s-%d.txt' % (args.tag, args.estimator, args.dim), 'w')

    for sigma_ in sigma_space:
        lists = []
        for lam_ in lam_space:
            loss = sess.run(l2_loss, feed_dict={ lam: lam_, sigma: sigma_ })
            txt = 'lam=%g, sigma=%g, loss=%g' % (lam_, sigma_, loss)
            print(txt)
            f.write(txt + '\n')
            lists.append(loss)
        f.flush()
        results.append(list)
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_data', type=int, default=128, help='data sample size.')
    parser.add_argument('--n_test', type=int, default=1024, help='test sample size.')
    parser.add_argument('--repeat', type=int, default=4)
    parser.add_argument('--dim', type=int, default=128, help='dimension.')
    parser.add_argument('--estimator', type=str, default='nu', 
            help='score estimator.', choices=['nu', 'landweber', 'tikhonov',
                'tikhonov_nystrom', 'spectral_cutoff', 'stein'])
    parser.add_argument('--kernel', type=str, default='curlfree_imq',
            help='matrix-valued kernel.', choices=['curlfree_imq',
                'curlfree_rbf', 'diagonal_imq', 'diagonal_rbf'])
    parser.add_argument('--subsample_rate', default=None, type=float,
            help='subsample rate used in the Nystrom approimation.')
    parser.add_argument('--tag', default='log', type=str)
    args = parser.parse_args(sys.argv[1:])
    with tf.Session() as sess:
        main(args, sess, args.repeat)
