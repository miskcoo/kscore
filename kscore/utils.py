#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf

def median_heuristic(x, y):
    # x: [..., n, d]
    # y: [..., m, d]
    # return: []
    n = tf.shape(x)[-2]
    m = tf.shape(y)[-2]
    x_expand = tf.expand_dims(x, -2)
    y_expand = tf.expand_dims(y, -3)
    pairwise_dist = tf.sqrt(tf.reduce_sum(tf.square(x_expand - y_expand), axis=-1))
    k = n * m // 2
    top_k_values = tf.nn.top_k(tf.reshape(pairwise_dist, [-1, n * m]), k=k).values
    kernel_width = tf.reshape(top_k_values[:, -1], tf.shape(x)[:-2])
    return tf.stop_gradient(kernel_width)

def random_choice(inputs, n_samples):
    """
    With replacement.
    Params:
      inputs (Tensor): Shape [n_states, n_features]
      n_samples (int): The number of random samples to take.
    Returns:
      sampled_inputs (Tensor): Shape [n_samples, n_features]
    """
    # (1, n_states) since multinomial requires 2D logits.
    uniform_log_prob = tf.expand_dims(tf.zeros(tf.shape(inputs)[0]), 0)

    ind = tf.random.categorical(uniform_log_prob, n_samples)
    ind = tf.squeeze(ind, 0, name="random_choice_ind")  # (n_samples,)

    return tf.gather(inputs, ind, name="random_choice")

def conjugate_gradient(operator,
                       rhs,
                       x=None,
                       tol=1e-4,
                       max_iter=40):
    '''From tensorflow/contrib/solvers/linear_equations.py'''

    cg_state = collections.namedtuple("CGState", ["i", "x", "r", "p", "gamma"])
  
    def stopping_criterion(i, state):
        return tf.logical_and(i < max_iter, tf.norm(state.r) > tol)
  
    def cg_step(i, state):
        z = operator.apply(state.p)
        alpha = state.gamma / tf.reduce_sum(state.p * z)
        x = state.x + alpha * state.p
        r = state.r - alpha * z
        gamma = tf.reduce_sum(r * r)
        beta = gamma / state.gamma
        p = r + beta * state.p
        return i + 1, cg_state(i + 1, x, r, p, gamma)
  
    n = operator.shape[1:]
    rhs = tf.expand_dims(rhs, -1)
    if x is None:
        x = tf.expand_dims(tf.zeros(n, dtype=rhs.dtype.base_dtype), -1)
        r0 = rhs
    else:
        x = tf.expand_dims(x, -1)
        r0 = rhs - operator.apply(x)

    p0 = r0
    gamma0 = tf.reduce_sum(r0 * p0)
    tol *= tf.norm(r0)
    i = tf.constant(0, dtype=tf.int32)
    state = cg_state(i=i, x=x, r=r0, p=p0, gamma=gamma0)
    _, state = tf.while_loop(stopping_criterion, cg_step, [i, state])
    return cg_state(
            state.i,
            x=tf.squeeze(state.x),
            r=tf.squeeze(state.r),
            p=tf.squeeze(state.p),
            gamma=state.gamma)
