#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

import argparse
import numpy as np
import tensorflow as tf
import zhusuan as zs
from tensorflow.contrib import layers
from tensorflow.python.client import timeline

from .utils import *
from .datasets import *

@zs.reuse('model')
def wae(observed, n, n_x, n_z, n_particles):
    with zs.BayesianNet(observed=observed) as model:
        z_mean = tf.zeros([n, n_z])
        z = zs.Normal('z', z_mean, std=1., group_ndims=1, n_samples=n_particles)
        lx_z = tf.layers.dense(z, units=256, activation=tf.nn.relu)
        lx_z = tf.layers.dense(lx_z, units=256, activation=tf.nn.relu)
        x = tf.layers.dense(lx_z, units=784, activation=None)
        zs.Bernoulli('x', x, group_ndims=1, dtype=tf.int32)
    return model, z, tf.sigmoid(x)

@zs.reuse('variational')
def q_net(x, n_z, n_particles):
    with zs.BayesianNet() as variational:
        x = tf.to_float(x)
        lz_x = tf.layers.dense(x, units=256, activation=tf.nn.relu)
        lz_x = tf.layers.dense(lz_x, units=256, activation=tf.nn.relu)
        z_mean = tf.layers.dense(lz_x, units=n_z, activation=None)
        z_logstd = tf.layers.dense(lz_x, units=n_z, activation=None)
        z = zs.Normal('z', z_mean, logstd=z_logstd, group_ndims=1, n_samples=n_particles)
    return z

def main(args):
    tf.set_random_seed(1237)
    np.random.seed(1234)

    # Load MNIST
    data_path = os.path.join('data', 'mnist.pkl.gz')
    x_train, t_train, x_valid, t_valid, x_test, t_test = \
        load_mnist_realval(data_path)
    x_train = np.vstack([x_train, x_valid])
    x_test = np.random.binomial(1, x_test, size=x_test.shape)
    n_x = x_train.shape[1]
    n_z = args.n_z
    lam = 1.0

    x_input = tf.placeholder(tf.float32, shape=[None, n_x], name='x')
    x = tf.to_int32(tf.random_uniform(tf.shape(x_input)) <= x_input)
    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, beta1=0.5)

    def build_graph(x):
        n = tf.shape(x)[0]
        x = tf.to_float(x)

        # qz_samples: [1, n, n_z]
        qz_samples = q_net(x, n_z, 1)
        # Use a single particle for the reconstruction term
        observed = { 'x': tf.to_int32(x), 'z': qz_samples }
        model, z, x_recons = wae(observed, n, n_x, n_z, 1)
        # log_pz: [1, n]
        log_pz = model.local_log_prob('z')
        # recons_err: [1, n]
        recons_err = tf.reduce_sum(tf.square(x_recons - x), axis=-1) ** 0.5
        loss_1 = tf.reduce_mean(recons_err - lam * log_pz)

        estimator = get_estimator(args)

        qzs = tf.squeeze(qz_samples, 0)
        estimator.fit(qzs)
        dlog_q = estimator.compute_gradients(qzs)
        entropy_surrogate = tf.reduce_mean(
            tf.reduce_sum(tf.stop_gradient(-dlog_q) * qzs, -1))
        cost = loss_1 - lam * entropy_surrogate
        model_params = tf.trainable_variables(scope="model") \
                + tf.trainable_variables(scope="variational")
        grads_and_vars = optimizer.compute_gradients(cost, var_list=model_params)

        return grads_and_vars, cost

    grads, eq_joint = build_graph(x)
    infer_op = optimizer.apply_gradients(grads)

    # Generate images
    n_gen = 100
    _, _, x_logits = wae({}, n_gen, n_x, n_z, 1)
    x_gen = tf.reshape(x_logits, [-1, 28, 28, 1])

    # Define training parameters
    learning_rate = 1e-4
    epochs = 1000
    batch_size = 64
    iters = x_train.shape[0] // batch_size
    save_image_freq = 100
    save_model_freq = 100
    result_path = "results/wae_mnist_relu_{}_{}_{}_{}".format(
        n_z, args.lam, args.estimator, args.kernel) \
                + time.strftime("_%Y%m%d_%H%M%S")

    saver = tf.train.Saver(max_to_keep=2)
    logger = setup_logger('wae_mnist', __file__, result_path)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        # Restore from the latest checkpoint
        ckpt_file = tf.train.latest_checkpoint(result_path)
        begin_epoch = 1
        if ckpt_file is not None:
            logger.info('Restoring model from {}...'.format(ckpt_file))
            begin_epoch = int(ckpt_file.split('.')[-2]) + 1
            saver.restore(sess, ckpt_file)

        for epoch in range(begin_epoch, epochs + 1):
            time_epoch = -time.time()
            np.random.shuffle(x_train)
            eq_joints = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                _, eq_joint_ = sess.run(
                    [infer_op, eq_joint],
                    feed_dict={x_input: x_batch,
                               learning_rate_ph: learning_rate},
                )

                eq_joints.append(eq_joint_)

            time_epoch += time.time()
            logger.info(
                'Epoch {} ({:.1f}s): cost = {}'
                .format(epoch, time_epoch, np.mean(eq_joints)))

            if epoch % save_image_freq == 0:
                logger.info('Saving images...')
                images = sess.run(x_gen)
                name = os.path.join(result_path,
                        "wae.epoch.{}.png".format(epoch))
                save_image_collections(images, name)

            if epoch % save_model_freq == 0:
                logger.info('Saving model...')
                save_path = os.path.join(result_path,
                        "wae.epoch.{}.ckpt".format(epoch))
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                saver.save(sess, save_path)
                logger.info('Done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_estimator_params(parser)
    parser.add_argument('--n_z', type=int, default=8, help='latent dimension.')
    args = parser.parse_args(sys.argv[1:])
    main(args)
