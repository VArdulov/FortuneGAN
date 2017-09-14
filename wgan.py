

import tensorflow as tf
import numpy as np


"""
    Implements the Improved WGAN algorithm given two network architectures.
    https://arxiv.org/pdf/1704.00028.pdf
"""
class WGAN(object):


    """
        generator_model:
            function that takes an input tensor of shape `[None]+generator_input_shape`
            and returns an output tensor whose 0th dimension is None

        critic_model:
            function that takes an input tensor of shape `[None]+critic_input_shape`
            and returns an output tensor whose 0th dimension is None
    """
    def __init__(self,
        generator_model,
        generator_input_shape,
        critic_model,
        critic_input_shape):

        self.generator_input_shape = generator_input_shape
        self.critic_input_shape = critic_input_shape

        generator_input_shape = [None] + generator_input_shape
        critic_input_shape = [None] + critic_input_shape

        self.generator_input_sym = tf.placeholder(tf.float32, shape=generator_input_shape, name='generator_input')
        self.generator_output_sym = generator_model(self.generator_input_sym)


        self.critic_input_sym = tf.placeholder_with_default(
            self.generator_output_sym,
            shape=critic_input_shape,
            name='critic_input'
        )
        self.critic_output_sym = critic_model(self.critic_input_sym)


        # Used for gradient normalization by the algorithm.
        self.epsilon_sym = tf.placeholder(tf.float32, shape=[None], name='gradient_interp')



    """
        Performs one step of training.
        Uses Algorithm 1 from https://arxiv.org/pdf/1704.00028.pdf.

        `num_critic_steps` is the number of critic training steps that should
        be taken per one generator training step

        next(sample_real_data) should yield a batch of 'real' data that will
        be passed to the critic

        next(sample_latent_variable) should yield a batch of 'fake' data that
        will be passed to the generator

        The shape of next(sample_real_data) and next(sample_latent_variable)
        should be the same!
    """
    def train(self, sess, num_critic_steps, sample_real_data, sample_latent_variable):
        for critic_step in range(num_critic_steps):
            # sample a batch of real data
            x_real = next(sample_real_data)

            # sample a batch of latent variables
            latent = next(sample_latent_variable)

            assert(x_real.shape[0] == latent.shape[0])
            batch_size = x_real.shape[0]

            # pick a batch of random epsilon in [0...1] (for gradient normalization)
            epsilon = np.random.rand(batch_size)

            # train the cirtic using the loss on the batch
            sess.run(self.train_critic, feed_dict={
                self.generator_input_sym : latent,
                self.critic_input_sym : x_real,
                self.epsilon_sym : epsilon
            })

        # sample a batch of latent variables again
        latent = next(sample_latent_variable)

        # train the generator using the loss on the batch
        sess.run(self.train_generator, feed_dict={
            self.generator_input_sym : latent
        })
