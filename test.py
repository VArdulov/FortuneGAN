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
    layer = relu_fc(layer, 30, 'layer2', reuse=reuse)
    output = fc(layer, 1, 'layer_out', reuse=reuse)
    return output


def critic_model(input_sym, reuse=False):
    layer = relu_fc(input_sym, 60, 'layer1', reuse=reuse)
    layer = relu_fc(layer, 60, 'layer2', reuse=reuse)
    output = fc(layer, 1, 'layer_out', reuse=reuse)
    return output

def sample_real_data(size):
    while True:
        yield np.random.randn(size, 1)

def draw_density_histogram(histogram, bin_width, max_height=1.0, num_rows=5):
    for p in np.linspace(max_height, 0, num_rows, endpoint=False):
        print('{:.3f}'.format(p), end='')
        for h in histogram:
            if h * bin_width >= p:
                print(u'   \u2588', end='')
            else:
                print('    ', end='')
        print()

    print('bins:', end='')
    for idx in range(20):
        print('{:4}'.format(idx), end='')
    print()

def draw_histogram(histogram, *, max_height, num_rows):
    draw_density_histogram(histogram, 1, max_height=max_height, num_rows=num_rows)

def test_wgan(wgan):
    real_hist, _ = np.histogram(np.random.randn(10000), bins=20, range=(-4, 4), density=True)


    generator_output = wgan.generator_output_sym.eval(feed_dict={
        wgan.generator_noise_sym : np.random.rand(10000, 1)
    })

    fake_hist, _ = np.histogram(generator_output, bins=20, range=(-4, 4), density=True)

    critic_inputs = np.expand_dims(np.linspace(-4, 4, 200), axis=1)
    critic_outputs = wgan.critic_output_real_sym.eval(feed_dict={
        wgan.critic_input_real_sym : critic_inputs
    })
    critic_hist = np.array([critic_outputs[range(10*a, 10*a+10)].mean() for a in range(20)])
    critic_hist -= critic_hist.min()
    critic_hist /= critic_hist.max()


    # Test for exploding weights.
    print("Number of nonfinite generator outputs: {}/10000".format(np.sum(np.logical_not(np.isfinite(generator_output)))))
    print("Number of nonfinite critic outputs: {}/200".format(np.sum(np.logical_not(np.isfinite(critic_outputs)))))


    print("Testing the WGAN's generator and critic...")
    print("Integral of real_hist is {}".format(
        (real_hist * (8.0 / 20.0)).sum()
    ))
    print("Integral of fake_hist is {}".format(
        (fake_hist * (8.0 / 20.0)).sum()
    ))

    print("Real distribution:")
    draw_density_histogram(real_hist, 8.0/20.0, max_height=0.3)
    print("Generated distribution:")
    draw_density_histogram(fake_hist, 8.0/20.0, max_height=0.3)
    print("Critic's histogram (low==fake, high==real):")
    draw_histogram(critic_hist, max_height=1.0, num_rows=5)


wgan = WGAN([1], generator_model, critic_model, gradient_penalty_factor=0)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # TODO: Initialize only WGAN variables.

    with sess.as_default():

        for i in range(10000):
            wgan.train(50, sample_real_data(20))

            if i % 1000 == 0:
                test_wgan(wgan)
