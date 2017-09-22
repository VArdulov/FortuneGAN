

import tensorflow as tf
import numpy as np


"""
    Implements the Improved WGAN algorithm given two network architectures.
    https://arxiv.org/pdf/1704.00028.pdf
"""
class WGAN(object):


    """
        generator_noise_shape:
            Shape of 1 noise input to the generator (no None entries).

        generator_model:
            Function that takes a tensor that represents a batch of noise
            and outputs a tuple:
                [0] a tensor that represents fake data
                [1] the list of variables created

        critic_model:
            Function that takes a tensor that represents a batch of data
            and outputs a tuple:
                [0] a tensor that represents the critic's predictions
                [1] the list of variables created

            This function may be called within a variable scope with reuse=True.

            The output should be a 1-dimensional vector whose length is the batch size.
    """
    def __init__(self,
        generator_model,
        generator_noise_shape,
        critic_model,
        gradient_penalty_factor=10):

        self.generator_noise_shape = generator_noise_shape

        generator_noise_shape = [None] + generator_noise_shape
        critic_input_shape = [None] + critic_input_shape

        with tf.variable_scope(name='generator', reuse=False):
            self.generator_noise_sym = tf.placeholder(tf.float32, shape=generator_noise_shape, name='generator_noise')
            self.generator_output_sym, self.generator_vars = generator_model(self.generator_noise_sym)


        with tf.variable_scope(name='critic', reuse=False):
            self.critic_input_real_sym = tf.placeholder(
                tf.float32,
                shape=self.generator_output_sym.get_shape(),
                name='critic_input_real'
            )
            self.critic_output_real_sym, self.critic_vars = critic_model(self.critic_input_sym)


        # Used for gradient normalization.
        self.epsilon_sym = tf.placeholder(tf.float32, shape=[None], name='gradient_interp')

        with tf.variable_scope(name='critic', reuse=True):
            self.critic_output_fake_sym, _ = critic_model(self.generator_output_sym)

            x_hat = self.epsilon_sym * self.critic_input_real_sym + (1 - self.epsilon_sym) * self.generator_output_sym
            self.critic_output_interp_sym, _ = critic_model(x_hat)



        interp_gradient = tf.gradients(self.critic_output_interp_sym, x_hat)[0]

        gradient_penalty = tf.norm(interp_gradient, axis=1) - 1
        gradient_penalty = gradient_penalty_factor * gradient_penalty * gradient_penalty

        self.critic_loss = (
            tf.reduce_mean( self.critic_output_fake_sym )
            - tf.reduce_mean( self.critic_output_real_sym )
            + tf.reduce_mean( gradient_penalty )
        )

        self.generator_loss = -tf.reduce_mean( self.critic_output_fake_sym )


        self.critic_optimizer = tf.train.AdamOptimizer(1e-3)
        self.generator_optimizer = tf.train.AdamOptimizer(1e-3)

        self.train_critic = critic_optimizer.minimize(self.critic_loss, var_list=self.critic_vars)
        self.train_generator = generator_optimizer.minimize(self.generator_loss, var_list=self.generator_vars)



    """
        Performs one step of training.
        Uses Algorithm 1 from https://arxiv.org/pdf/1704.00028.pdf.

        `num_critic_steps` is the number of critic training steps that should
        be taken per one generator training step.

        next(sample_real_data) should yield a batch of 'real' data that will
        be passed to the critic. It is assumed that this always returns a batch
        of the same size, i.e. the 0th dimension should always have the same size.
    """
    def train(self, sess, num_critic_steps, sample_real_data):
        for critic_step in range(num_critic_steps):
            # sample a batch of real data
            x_real = next(sample_real_data)
            batch_size = x_real.shape[0]


            # sample a batch of noise
            noise = np.random.rand(batch_size, *self.generator_noise_shape)

            # pick a batch of random epsilon in [0...1] (for gradient normalization)
            epsilon = np.random.rand(batch_size)

            # train the cirtic using the loss on the batch
            sess.run(self.train_critic, feed_dict={
                self.generator_noise_sym : noise,
                self.critic_input_real_sym : x_real,
                self.epsilon_sym : epsilon
            })

        # sample a batch of noise
        noise = np.random.rand(batch_size, *self.generator_noise_shape)

        # train the generator using the loss on the batch
        sess.run(self.train_generator, feed_dict={
            self.generator_noise_sym : noise
        })
