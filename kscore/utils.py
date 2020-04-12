#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

    ind = tf.multinomial(uniform_log_prob, n_samples)
    ind = tf.squeeze(ind, 0, name="random_choice_ind")  # (n_samples,)

    return tf.gather(inputs, ind, name="random_choice")
