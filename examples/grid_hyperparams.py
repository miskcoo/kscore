#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import tensorflow as tf
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
    elif args.estimator == 'tikhonov':
        estimator = kscore.estimators.Tikhonov(lam=lam, use_cg=True, kernel=kernel)
    elif args.estimator == 'spectral_cutoff_percent':
        estimator = kscore.estimators.SpectralCutoff(lam=None, keep_rate=lam, kernel=kernel)
    else:
        estimator = estimator_dicts[args.estimator](lam=lam, kernel=kernel)
    return estimator

def main(args, sess, repeat=4):
    tf.compat.v1.set_random_seed(1234)
    np.random.seed(1234)

    dim = args.dim
    losses = []
    lam = tf.placeholder(tf.float32, shape=[])
    sigma = tf.placeholder(tf.float32, shape=[])

    grid_vertices = np.random.randint(low=0, high=2, size=[dim, dim]).astype(np.float32)
    grid_vertices *= args.cube_length

    for _ in range(repeat):
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

    if args.sigma_scale == 'lin':
        sigma_space = np.linspace(args.sigma_low, args.sigma_high, args.bins)
    else:
        sigma_space = np.logspace(args.sigma_low, args.sigma_high, args.bins)

    if args.estimator == 'spectral_cutoff_percent':
        lam_space = np.linspace(args.lam_low, args.lam_high, args.bins)
    else:
        lam_space = np.logspace(args.lam_low, args.lam_high, args.bins)

    results = []
    f = open('%s-%s-%s-%d.txt' % (args.tag, args.estimator, args.kernel, args.dim), 'w')

    f.write(' '.join(map(str, [args.estimator, args.repeat, args.sigma_low,
        args.sigma_high, args.lam_low, args.lam_high, args.bins, args.dim, args.sigma_scale])))
    f.write('\n')

    for sigma_ in sigma_space:
        lists = []
        for lam_ in lam_space:
            loss = sess.run(l2_loss, feed_dict={ lam: lam_, sigma: sigma_ })
            txt = 'lam=%g, sigma=%g, loss=%g' % (lam_, sigma_, loss)
            print(txt)
            f.write(txt + '\n')
            lists.append(loss)
        f.flush()
        results.append(lists)
    f.close()

    if args.show_figure:
        import matplotlib.pyplot as plt
        Z_clip = np.minimum(results, np.min(results) * 3.0)
        Z_clip = np.log(Z_clip)
        if args.estimator == 'spectral_cutoff_percent':
            plt.xlabel('keep rate of eigenvalues')
        else:
            plt.xlabel('regularization parameter')
            plt.xscale('log')
        if args.sigma_scale == 'log':
            plt.yscale('log')
        plt.ylabel('kernel bandwidth')
        ct = plt.contour(lam_space, sigma_space, Z_clip, 10)
        plt.clabel(ct, fontsize=7, inline=1)
        idx = np.argmin(results)
        argmin_x, argmin_y = lam_space[idx % args.bins], sigma_space[idx // args.bins]
        print("Minimal loss = %.4f, attained when lam = %.4g, sigma = %.4g" % (np.min(results), argmin_x, argmin_y))
        plt.scatter([argmin_x], [argmin_y], marker='x', s=70)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_data', type=int, default=128, help='data sample size.')
    parser.add_argument('--n_test', type=int, default=1024, help='test sample size.')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--dim', type=int, default=32, help='dimension.')
    parser.add_argument('--estimator', type=str, default='nu', 
            help='score estimator.', choices=['nu', 'landweber', 'tikhonov',
                'tikhonov_nystrom', 'spectral_cutoff', 'stein', 'spectral_cutoff_percent'])
    parser.add_argument('--kernel', type=str, default='curlfree_imq',
            help='matrix-valued kernel.', choices=['curlfree_imq',
                'curlfree_rbf', 'diagonal_imq', 'diagonal_rbf'])
    parser.add_argument('--subsample_rate', default=None, type=float,
            help='subsample rate used in the Nystrom approimation.')
    parser.add_argument('--tag', default='log', type=str)
    parser.add_argument('--bins', default=10, type=int)
    parser.add_argument('--lam_low', default=-6, type=float)
    parser.add_argument('--lam_high', default=-2, type=float)
    parser.add_argument('--sigma_low', default=5, type=float)
    parser.add_argument('--sigma_high', default=40, type=float)
    parser.add_argument('--sigma_scale', default='lin', type=str, choices=['lin', 'log'])
    parser.add_argument('--cube_length', default=1, type=float)
    parser.add_argument('--show_figure', default=1, type=int)

    args = parser.parse_args(sys.argv[1:])
    with tf.Session() as sess:
        main(args, sess, args.repeat)
