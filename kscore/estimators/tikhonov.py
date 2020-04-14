#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf

from kscore.utils import random_choice, conjugate_gradient
from kscore.kernels import CurlFreeIMQ
from .base import Base

class Tikhonov(Base):
    def __init__(self,
                 lam,
                 kernel=CurlFreeIMQ(),
                 truncated_tikhonov=False,
                 subsample_rate=None,
                 use_cg=True,
                 tol_cg=1.0e-4,
                 maxiter_cg=40,
                 dtype=tf.float32):
        super().__init__(lam, kernel, dtype)
        self._use_cg = use_cg
        self._tol_cg = tol_cg
        self._subsample_rate = subsample_rate
        self._maxiter_cg = maxiter_cg
        self._truncated_tikhonov = truncated_tikhonov

    def fit(self, samples, kernel_hyperparams=None):
        if self._subsample_rate is None:
            return self._fit_exact(samples, kernel_hyperparams)
        else:
            return self._fit_subsample(samples, kernel_hyperparams)

    def _compute_energy(self, x):
        d = tf.shape(x)[-1]
        if self._subsample_rate is None and not self._truncated_tikhonov:
            Kxq, div_xq = self._kernel.kernel_energy(x, self._samples,
                    kernel_hyperparams=self._kernel_hyperparams)
            div_xq = tf.reduce_mean(div_xq, axis=-1) / self._lam
            Kxq = tf.reshape(Kxq, [tf.shape(x)[-2], -1])
            energy = tf.matmul(Kxq, self._coeff)
            energy = tf.reshape(energy, [-1]) - div_xq
        else:
            Kxq = self._kernel.kernel_energy(x, self._samples,
                    kernel_hyperparams=self._kernel_hyperparams, compute_divergence=False)
            Kxq = tf.reshape(Kxq, [tf.shape(x)[-2], -1])
            energy = -tf.matmul(Kxq, self._coeff)
            energy = tf.reshape(energy, [-1])
        return energy

    def compute_gradients(self, x):
        d = tf.shape(x)[-1]
        if self._subsample_rate is None and not self._truncated_tikhonov:
            Kxq_op, div_xq = self._kernel.kernel_operator(x, self._samples,
                    kernel_hyperparams=self._kernel_hyperparams)
            div_xq = tf.reduce_mean(div_xq, axis=-2) / self._lam
            grads = Kxq_op.apply(self._coeff)
            grads = tf.reshape(grads, [-1, d]) - div_xq
        else:
            Kxq_op = self._kernel.kernel_operator(x, self._samples,
                    kernel_hyperparams=self._kernel_hyperparams, compute_divergence=False)
            grads = -Kxq_op.apply(self._coeff)
            grads = tf.reshape(grads, [-1, d])
        return grads

    def _fit_subsample(self, samples, kernel_hyperparams=None):
        if kernel_hyperparams is None:
            kernel_hyperparams = self._kernel.heuristic_hyperparams(samples, samples)
        self._kernel_hyperparams = kernel_hyperparams

        M = tf.shape(samples)[-2]
        N = tf.cast(tf.cast(M, self._dtype) * self._subsample_rate, tf.int32)
        d = tf.shape(samples)[-1]

        subsamples = random_choice(samples, N)
        Knn_op = self._kernel.kernel_operator(subsamples, subsamples, 
                kernel_hyperparams=kernel_hyperparams, compute_divergence=False)
        Knm_op, K_div = self._kernel.kernel_operator(subsamples, samples,
                kernel_hyperparams=kernel_hyperparams)
        self._samples = subsamples

        if self._use_cg:
            def apply_kernel(v):
                return Knm_op.apply(Knm_op.apply_adjoint(v)) / tf.cast(M, self._dtype) \
                        + self._lam * Knn_op.apply(v)

            linear_operator = collections.namedtuple(
                "LinearOperator", ["shape", "dtype", "apply", "apply_adjoint"])
            Kcg_op = linear_operator(
                shape=Knn_op.shape,
                dtype=Knn_op.dtype,
                apply=apply_kernel,
                apply_adjoint=apply_kernel,
            )
            H_dh = tf.reduce_mean(K_div, axis=-2)
            H_dh = tf.reshape(H_dh, [N * d])
            conj_ret = conjugate_gradient(
                    Kcg_op, H_dh, max_iter=self._maxiter_cg, tol=self._tol_cg)
            self._coeff = tf.reshape(conj_ret.x, [N * d, 1])
        else:
            Knn = Knn_op.kernel_matrix(flatten=True)
            Knm = Knm_op.kernel_matrix(flatten=True)
            K_inner = tf.matmul(Knm, Knm, transpose_b=True) / tf.cast(M, self._dtype) + self._lam * Knn
            H_dh = tf.reduce_mean(K_div, axis=-2)

            if self._kernel.kernel_type() == 'diagonal':
                K_inner += 1.0e-7 * tf.eye(N)
                H_dh = tf.reshape(H_dh, [N, d])
            else:
                # The original Nystrom KEF estimator (Sutherland et al., 2018).
                # Adding the small identity matrix is necessary for numerical stability.
                K_inner += 1.0e-7 * tf.eye(N * d)
                H_dh = tf.reshape(H_dh, [N * d, 1])
            self._coeff = tf.reshape(tf.linalg.solve(K_inner, H_dh), [N * d, 1])

    def _fit_exact(self, samples, kernel_hyperparams=None):
        # samples: [M, d]
        if kernel_hyperparams is None:
            kernel_hyperparams = self._kernel.heuristic_hyperparams(samples, samples)
        self._kernel_hyperparams = kernel_hyperparams
        self._samples = samples

        M = tf.shape(samples)[-2]
        d = tf.shape(samples)[-1]

        K_op, K_div = self._kernel.kernel_operator(samples, samples,
                kernel_hyperparams=kernel_hyperparams)

        if self._use_cg:
            if self._truncated_tikhonov:
                def apply_kernel(v):
                    return K_op.apply(K_op.apply(v) / tf.cast(M, self._dtype) + self._lam * v)
            else:
                def apply_kernel(v):
                    return K_op.apply(v) + tf.cast(M, self._dtype) * self._lam * v

            linear_operator = collections.namedtuple(
                "LinearOperator", ["shape", "dtype", "apply", "apply_adjoint"])
            Kcg_op = linear_operator(
                shape=K_op.shape,
                dtype=K_op.dtype,
                apply=apply_kernel,
                apply_adjoint=apply_kernel,
            )
            H_dh = tf.reduce_mean(K_div, axis=-2)
            H_dh = tf.reshape(H_dh, [M * d]) / self._lam
            conj_ret = conjugate_gradient(
                    Kcg_op, H_dh, max_iter=self._maxiter_cg, tol=self._tol_cg)
            self._coeff = tf.reshape(conj_ret.x, [M * d, 1])
        else:
            K = K_op.kernel_matrix(flatten=True)
            H_dh = tf.reduce_mean(K_div, axis=-2)
            if self._kernel.kernel_type() == 'diagonal':
                identity = tf.eye(M)
                H_shape = [M, d]
            else:
                identity = tf.eye(M * d)
                H_shape = [M * d, 1]

            if self._truncated_tikhonov:
                # The Nystrom version of KEF with full samples.
                # See Example 3.6 for more details.
                K = tf.matmul(K, K) / tf.cast(M, self._dtype) \
                        + self._lam * K + 1.0e-7 * identity
            else:
                # The original KEF estimator (Sriperumbudur et al., 2017).
                K += tf.cast(M, self._dtype) * self._lam * identity
            H_dh = tf.reshape(H_dh, H_shape) / self._lam
            self._coeff = tf.reshape(tf.linalg.solve(K, H_dh), [M * d, 1])

