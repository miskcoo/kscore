#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from kscore import *

class TestEstimator(tf.test.TestCase):
    def setUp(self):
        tf.set_random_seed(1234)
        heuristic = lambda x, y: kernels.median_heuristic(x, y) * 1.5
        self.diagonal_kernel = kernels.DiagonalIMQ(heuristic_hyperparams=heuristic)
        self.curlfree_kernel = kernels.CurlFreeIMQ(heuristic_hyperparams=heuristic)

    def _test(self, sess, estimator, l2_bound, cos_bound):
        def _test0(kernel_hyperparams):
            x_train = tf.random.normal([256, 2])
            x_test = tf.random.normal([256, 2])
            y_ans = -0.5 * x_test

            estimator.fit(x_train, kernel_hyperparams=kernel_hyperparams)
            y_test = estimator.compute_gradients(x_test)

            l2_dist = tf.norm(y_test - y_ans, axis=-1) 
            l2_dist /= tf.norm(y_ans, axis=-1) + 1
            l2_dist = tf.reduce_mean(l2_dist)

            cos_dist = tf.reduce_sum(y_test * y_ans, axis=-1) 
            cos_dist /= tf.norm(y_ans, axis=-1) * tf.norm(y_test, axis=-1)
            cos_dist = tf.reduce_mean(cos_dist)

            l2_dist, cos_dist = sess.run([l2_dist, cos_dist])
            self.assertLess(l2_dist, l2_bound)
            self.assertGreater(cos_dist, cos_bound)

        _test0(3.0)
        _test0(None)

    def _test_tikhonov(self, subsample_rate, truncated_tikhonov):
        est_curlfree_cg = estimators.Tikhonov(
            lam=0.1, 
            kernel=self.curlfree_kernel,
            truncated_tikhonov=truncated_tikhonov,
            subsample_rate=subsample_rate,
            use_cg=True,
            maxiter_cg=100,
        )

        est_curlfree_nocg = estimators.Tikhonov(
            lam=0.1, 
            kernel=self.curlfree_kernel,
            truncated_tikhonov=truncated_tikhonov,
            subsample_rate=subsample_rate,
            use_cg=False
        )

        est_diagonal_cg = estimators.Tikhonov(
            lam=0.1, 
            kernel=self.diagonal_kernel,
            truncated_tikhonov=truncated_tikhonov,
            subsample_rate=subsample_rate,
            use_cg=True,
            maxiter_cg=100,
        )

        est_diagonal_nocg = estimators.Tikhonov(
            lam=0.1, 
            kernel=self.diagonal_kernel,
            truncated_tikhonov=truncated_tikhonov,
            subsample_rate=subsample_rate,
            use_cg=False
        )

        with self.session() as sess:
            self._test(sess, est_curlfree_cg, 0.25, 0.95)
            self._test(sess, est_diagonal_cg, 0.25, 0.95)
            self._test(sess, est_curlfree_nocg, 0.25, 0.95)
            self._test(sess, est_diagonal_nocg, 0.25, 0.95)

    def test_tikhonov(self):
        self._test_tikhonov(None, False)

    def test_tikhonov_nystrom(self):
        self._test_tikhonov(0.7, False)
        self._test_tikhonov(0.7, True)

    def test_landweber(self):
        est_curlfree = estimators.Landweber(
            lam=0.001, 
            iternum=None,
            kernel=self.curlfree_kernel,
        )

        est_diagonal = estimators.Landweber(
            lam=0.1, 
            iternum=None,
            kernel=self.diagonal_kernel,
        )

        with self.session() as sess:
            self._test(sess, est_curlfree, 0.50, 0.95)
            self._test(sess, est_diagonal, 0.50, 0.95)


        est_curlfree = estimators.Landweber(
            lam=None,
            iternum=1000,
            kernel=self.curlfree_kernel,
        )

        est_diagonal = estimators.Landweber(
            lam=None, 
            iternum=10,
            kernel=self.diagonal_kernel,
        )

        with self.session() as sess:
            self._test(sess, est_curlfree, 0.50, 0.95)
            self._test(sess, est_diagonal, 0.50, 0.95)

    def test_nu_method(self):
        def _test(nu, lam, iternum):
            est_curlfree = estimators.NuMethod(
                lam=lam, 
                iternum=iternum,
                nu=nu,
                kernel=self.curlfree_kernel,
            )

            est_diagonal = estimators.NuMethod(
                lam=lam, 
                iternum=iternum,
                nu=nu,
                kernel=self.diagonal_kernel,
            )

            with self.session() as sess:
                self._test(sess, est_curlfree, 0.25, 0.95)
                self._test(sess, est_diagonal, 0.25, 0.95)

        _test(1.0, None, 5)
        _test(1.0, 0.1, None)
        _test(2.0, None, 5)
        _test(2.0, 0.1, None)

    def test_spectral_cutoff(self):
        def _test(lam, keep_rate):
            est_curlfree = estimators.SpectralCutoff(
                lam=lam, 
                keep_rate=keep_rate,
                kernel=self.curlfree_kernel,
            )

            est_diagonal = estimators.SpectralCutoff(
                lam=lam, 
                keep_rate=keep_rate,
                kernel=self.diagonal_kernel,
            )

            with self.session() as sess:
                # This is sensitive to parameters
                self._test(sess, est_curlfree, 0.6, 0.90)
                self._test(sess, est_diagonal, 0.6, 0.90)

        _test(0.005, None)

    def test_stein(self):
        est_diagonal = estimators.Stein(
            lam=0.001,
            kernel=self.diagonal_kernel,
        )

        with self.session() as sess:
            self._test(sess, est_diagonal, 0.3, 0.95)
