import os
import math
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.keras.layers as kl
import numpy as np

import time

np.random.seed(1)
tf.set_random_seed(1)

## Training parameters
batch_size = 64
update_freq = 1
y = 0.99
startE = 1
endE = 0.1
annealing_steps = 10000.0
num_episodes = 10000
pre_train_steps = 1000
load_model = False
path = "testing/"
h_size = 256
tau = 0.001

already_travelled = []


class QPathFinding:

    def __init__(self, h_size):
        self.scalarInput = tf.placeholder(shape=[None,256], dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 16, 16, 1])

        self.conv1 = slim.conv2d(inputs = self.imageIn, num_outputs = 16, \
                                 kernel_size = [2, 2], stride = [1, 1], \
                                 padding = 'VALID', biases_initializer=None)
        
        self.conv2 = slim.conv2d(inputs = self.conv1, num_outputs = 32, \
                                 kernel_size = [3, 3], stride = [2, 2], \
                                 padding = 'VALID', biases_initializer=None)

        self.conv3 = slim.conv2d(inputs = self.conv2, num_outputs = 64, \
                                 kernel_size = [2, 2], stride = [1, 1], \
                                 padding = 'VALID', biases_initializer=None)

        self.conv4 = slim.conv2d(inputs = self.conv3, num_outputs = h_size/4, \
                                 kernel_size = [3, 3], stride = [3, 3], \
                                 padding = 'VALID')

        self.streamAC, self.streamVC = tf.split(self.conv4, 2, 1)
        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.AW = tf.Variable(xavier_init([h_size//2, 4])) # 4 is number of environment actions
        self.VW = tf.Variable(xavier_init([h_size//2, 1]))
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)

        self.Qout = self.Value = tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
        self.predict = tf.argmax(self.Qout, 1)

        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, 4, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)



class experience_buffer():
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])


def process_state(states):
    return np.reshape(states, [256])


def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) \
                        + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder


def update_target(op_holder, sess):
    for op in op_holder:
        sess.run(op)


def get_row(idx, dim):
    return int(idx / dim)


def get_col(idx, dim):
    return idx % dim

def get_reward(start, end, moved, optimal_path, neighbors, new_dist):

    dist = len(optimal_path[0])-1
    
    if dist < 2: # If within interaction distance
        return 5
    
    if new_dist is None:
        return 0
    elif len(optimal_path[0]) == new_dist:
        return 0
    elif len(optimal_path[0]) < new_dist:
        return -1
    return 1


##    global already_travelled
##    path, dim = optimal_path
##    optimal_move = path[1]
##    optimal_x = get_row(optimal_move, dim)
##    optimal_y = get_col(optimal_move, dim)
##
##    result = 0
##
##    # If made the optimal move for
##    if (optimal_x, optimal_y) == (start[0], start[1]):
##        result = 3
##
##    # Encourage being near farm plots
##    # if "brown_shulker_box" in neighbors:
##    #     result += 0.08 * len([n for n in neighbors if n == "brown_shulker_box"])
##    # else:
##    #     result -= 0.08 * len([n for n in neighbors if not n == "brown_shulker_box"])
##
##    dist = len(path)-1
##    if dist < 2: # If within interaction distance
##        return result + 5
##    result -= dist * 0.08
##    if moved == -1: # If made an invalid move
##        result -= 5
##    else:
##        result -= 0.04
##    if start in already_travelled: # If has already been to block
##        result -= 0.5
##    else:
##        already_travelled.append(start)
##    return result


def reset_already_travelled():
    global already_travelled
    already_travelled = []



