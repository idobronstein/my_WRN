import numpy as np
import tensorflow as tf

## TensorFlow helper functions

WEIGHT_DECAY_KEY = 'WEIGHT_DECAY'

def _relu(x, leakness=0.0, name=None):
    if leakness > 0.0:
        name = 'lrelu' if name is None else name
        return tf.maximum(x, x*leakness, name='lrelu')
    else:
        name = 'relu' if name is None else name
        return tf.nn.relu(x, name='relu')

def _conv(x, filter_size, out_channel, strides, pad='SAME', name='conv', init_kernel=None):
    in_shape = x.get_shape()
    with tf.variable_scope(name):
        if init_kernel is None:
            kernel = tf.get_variable('kernel', [filter_size, filter_size, in_shape[3], out_channel],
                            tf.float32, initializer=tf.random_normal_initializer(
                                stddev=np.sqrt(2.0/filter_size/filter_size/out_channel)))
        else: 
            kernel = tf.get_variable('kernel', initializer=init_kernel)
        if kernel not in tf.get_collection(WEIGHT_DECAY_KEY):
            tf.add_to_collection(WEIGHT_DECAY_KEY, kernel)
            # print('\tadded to WEIGHT_DECAY_KEY: %s(%s)' % (kernel.name, str(kernel.get_shape().as_list())))
        conv = tf.nn.conv2d(x, kernel, [1, strides, strides, 1], pad)
    return conv

def _fc(x, out_dim, name='fc', init_params=[]):
    with tf.variable_scope(name):
        if len(init_params) == 0:
            w = tf.get_variable('weights', [x.get_shape()[1], out_dim],
                            tf.float32, initializer=tf.random_normal_initializer(
                                stddev=np.sqrt(1.0/out_dim)))
            b = tf.get_variable('biases', [out_dim], tf.float32,
                                initializer=tf.constant_initializer(0.0))
        else:
            w = tf.get_variable('weights', initializer=init_params[0])
            b = tf.get_variable('biases', initializer=init_params[1])
        if w not in tf.get_collection(WEIGHT_DECAY_KEY):
            tf.add_to_collection(WEIGHT_DECAY_KEY, w)
        fc = tf.nn.bias_add(tf.matmul(x, w), b)
    return fc


def _bn(x, is_train, global_step=None, name='bn', init_params=[]):
    moving_average_decay = 0.9

    with tf.variable_scope(name):
        decay = moving_average_decay

        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
        if len(init_params) == 0:
            mu = tf.get_variable('mu', batch_mean.get_shape(), tf.float32,
                            initializer=tf.zeros_initializer, trainable=False)
            sigma = tf.get_variable('sigma', batch_var.get_shape(), tf.float32,
                            initializer=tf.ones_initializer, trainable=False)
            beta = tf.get_variable('beta', batch_mean.get_shape(), tf.float32,
                            initializer=tf.zeros_initializer)
            gamma = tf.get_variable('gamma', batch_var.get_shape(), tf.float32,
                            initializer=tf.ones_initializer)
        else:
            mu = tf.get_variable('mu', initializer=init_params[0])
            sigma = tf.get_variable('sigma', initializer=init_params[1])
            beta = tf.get_variable('beta', initializer=init_params[2])
            gamma = tf.get_variable('gamma', initializer=init_params[3])
        # BN when training
        update = 1.0 - decay
        update_mu = mu.assign_sub(update*(mu - batch_mean))
        update_sigma = sigma.assign_sub(update*(sigma - batch_var))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mu)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_sigma)
        mean, var = tf.cond(is_train, lambda: (batch_mean, batch_var),
                            lambda: (mu, sigma))
        bn = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)
    return bn


## Other helper functions



