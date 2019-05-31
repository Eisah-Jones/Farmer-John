import os
import math
import time
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.keras.layers as kl
import numpy as np
import data_writer as dw
import AC_Network as ac

class FarmingNetwork:
    def __init__(self, num_actions = 32, load_model = False, model_path = 'default_checkpoint', exploration = 'e-greedy'):
        self.exploration = exploration
        self.model_path = model_path
        self.load_model = load_model

        # Hyper parameters
        self.layer_size = 64
        self.batch_size = 5
        self.learning_rate = 0.0001
        self.gamma = 0.99
        self.input_size = 32

        self.observations = tf.placeholder(tf.float32, [None, self.input_size])
        self.W1 = tf.get_variable('W1', shape=[self.input_size, self.layer_size],
                                  initializer=tf.contrib.layers.xavier_initializer())
        self.L1 = tf.nn.relu(tf.matmul(self.observations, self.W1))
        self.W2 = tf.get_variable('W2', shape=[self.layer_size, 1],
                                  initializer=tf.contrib.layers.xavier_initializer())
        self.score = tf.matmul(self.L1, self.W2)
        self.probability = tf.nn.sigmoid(self.score)

        self.tvars = tf.trainable_variables()
        self.input_y = tf.placeholder(tf.float32, [None, 1])
        self.advantages = tf.placeholder(tf.float32)

        self.log_likelihood = tf.log(self.input_y*(self.input_y - self.probability) +
                                     (1 - self.input_y)*(self.input_y + probability))
        self.loss = -tf.reduce_mean(self.log_likelihood * self.advantages)
        self.newGrads = tf.gradients(self.loss, self.tvars)

        adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.W1Grad = tf.placeholder(tf.float32)
        self.W2Grad = tf.placeholder(tf.float32)
        self.batchGrad = [W1Grad, W2Grad]
        self.updateGrads = adam.apply_gradients(zip(batchGrads, tvars))
        

    def test(self):
        pass


    def get_reward(self, state):
        pass


    def discount_rewards(r):
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(xrange(0, r.size)):
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r


    def action_to_dest(self, a):
        pass

    def get_decision(self, sess):
        if self.exploration == 'e-greedy':
            if random.random() < self.e:
                return random.randint(0, self.num_actions-1)
            else:
                return sess.run(self.chosen_action)
        return -1


    def save_session(self, sess, path):
        self.saver.save(sess, path)






