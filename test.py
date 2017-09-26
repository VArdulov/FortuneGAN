import tensorflow as tf
import numpy as np

from wgan import WGAN

################################################################################
"""
    Some functions to assist the definition of a basic neural network.
"""
def relu_fc(bottom, size, name, reuse=False):
    in_size = bottom.get_shape()[-1]
    with tf.variable_scope(name, reuse=reuse):
        if not reuse:
            weights = tf.get_variable('weights', dtype=tf.float32, shape=[in_size,size], initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable('biases', dtype=tf.float32, shape=[size], initializer=tf.contrib.layers.xavier_initializer())
        else:
            weights = tf.get_variable('weights')
            biases = tf.get_variable('biases')
        return tf.nn.relu_layer(bottom, weights, biases, name=name)

def fc(bottom, size, name, reuse=False):
    in_size = bottom.get_shape()[-1]
    with tf.variable_scope(name, reuse=reuse):
        if not reuse:
            weights = tf.get_variable('weights', dtype=tf.float32, shape=[in_size,size], initializer=tf.contrib.layers.variance_scaling_initializer())
            biases = tf.get_variable('biases', dtype=tf.float32, shape=[size], initializer=tf.contrib.layers.variance_scaling_initializer())
        else:
            weights = tf.get_variable('weights')
            biases = tf.get_variable('biases')
        return tf.nn.bias_add(tf.matmul(bottom, weights), biases, name=name)
################################################################################


def generator_model(noise_sym, reuse=False):
    layer = relu_fc(noise_sym, 30, 'layer1', reuse=reuse)
    output = fc(layer, 1, 'layer_out', reuse=reuse)
    return output


def critic_model(input_sym, reuse=False):
    layer = relu_fc(input_sym, 30, 'layer1', reuse=reuse)
    output = fc(layer, 1, 'layer_out', reuse=reuse)
    return output

def sample_real_data(size):
    while True:
        yield np.random.randn(size, 1)

def test_generator(generator_tensor):
    real_hist, _ = np.histogram(np.random.randn(10000), bins=20, range=(-4, 4), density=True)
    fake_hist, _ = np.histogram(generator_tensor.eval(feed_dict={
        wgan.generator_noise_sym : np.random.rand(10000, 1)
    }), bins=20, range=(-4, 4), density=True)

    for hist, mssg in [(real_hist, 'real'), (fake_hist, 'fake')]:
        print(mssg)
        for p in [1, .8, .6, .4, .2]:
            print('{:.3f}'.format(p), end='')
            for h in hist:
                if h * (20.0/8.0) >= p:
                    print(u'   \u2588', end='')
                else:
                    print('    ', end='')
            print()

        print('bins:', end='')
        for idx in range(20):
            print('{:4}'.format(idx), end='')
        print()


wgan = WGAN([1], generator_model, critic_model)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # TODO: Initialize only WGAN variables.

    with sess.as_default():

        for i in range(10000):
            wgan.train(10, sample_real_data(20))

            if i % 1000 == 0:
                test_generator(wgan.generator_output_sym)
