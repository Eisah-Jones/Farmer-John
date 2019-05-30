import os
import math
import time
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.keras.layers as kl
import numpy as np
import dueling_network as dn
import data_writer as dw


class PathfindingNetwork:
    def __init__(self, load_model = False, model_path = 'default_checkpoint', exploration = 'e-greedy'):
        self.batch_size = 64
        self.update_freq = 1
        self.y = 0.99
        self.startE = 1
        self.endE = 0.1
        self.annealing_steps = 1000000.0
        self.num_episodes = 50000
        self.pre_train_steps = 100000
        self.load_model = load_model
        self.model_path = 'data/pathfinding_network/' + model_path
        self.exploration = exploration
        self.tau = 0.001
        self.h_size = 1024

        self.mainQN = dn.DuelingNetwork(self.h_size, 4)
        self.targetQN = dn.DuelingNetwork(self.h_size, 4)

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.trainables = tf.trainable_variables()
        self.targetOps = updateTargetGraph(self.trainables, self.tau)
        self.networkBuffer = experience_buffer()
        self.e = self.startE
        self.stepDrop = (self.startE - self.endE) / self.annealing_steps
        self.log = dw.DataWriter('pathfinding_train_data')
        self.log.create_csv_record()
        
        self.check_path()


    def train(self, dest):
        trainBatch = self.networkBuffer.sample(self.batch_size, dest)
        Q1 = sess.run(self.mainQN.predict, feed_dict={self.mainQN.scalarInput:np.vstack(trainBatch[:,3])})
        Q2 = sess.run(self.targetQN.Qout, feed_dict={self.targetQN.scalarInput:np.vstack(trainBatch[:,3])})
        end_multiplier = -(trainBatch[:, 4] - 1)
        doubleQ = Q2[range(self.batch_size), Q1]
        targetQ = trainBatch[:,2] + (self.y*doubleQ*end_multiplier)
        _ = sess.run(self.mainQN.updateModel, \
            feed_dict={self.mainQN.scalarInput:np.vstack(trainBatch[:,0]), \
                       self.mainQN.targetQ:targetQ, self.mainQN.actions:trainBatch[:,1]})
        update_target(targetOps, sess)


    def test(self, sess, s):
        return sess.run(self.mainQN.predict, feed_dict={self.mainQN.scalarInput:[s]})[0]


    def get_action(self, s, total_steps):
        if self.exploration == 'e-greedy':
            if random.random() < self.e or (total_steps < self.pre_train_steps and not self.load_model):
                a = np.random.randint(0, 3)
            else:
                a = sess.run(self.mainQN.predict, feed_dict={self.mainQN.scalarInput: [s]})[0]
            return a


    def get_reward(self, start, end, moved, optimal_path, new_dist):
        reward = 0
        path, dim = optimal_path
        current_dist = new_dist - 1

        if current_dist <= 1:
            return 100

        reward -= current_dist * 0.35
        
        if len(path) <= new_dist:
            reward -= 20
        else:
            reward += 20

        if moved == -1:
            reward -= 10
        else:
            reward -+ 1

        return reward


    def decrease_epsilon(self):
        if self.e > self.endE:
            self.e -= self.stepDrop


    def add_episode_experience(self, experience):
        self.networkBuffer.add(experience)


    def save_session(self, sess, path):
        self.saver.save(sess, path)


    def check_path(self):
        if not os.path.exists('data/pathfinding_network'):
            os.mkdir('data/pathfinding_network')

        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
            


class experience_buffer():
    def __init__(self, buffer_size = 100000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size, dest):
        dest_buffer = []
        for experience in self.buffer:
            if experience[-1] == dest:
                dest_buffer.append(experience)

        if len(dest_buffer) < size:
            dest_buffer.extend(list(random.sample(self.buffer, size-len(dest_buffer)+1)))
            
        return np.reshape(np.array(random.sample(dest_buffer, size)), [size, 6])


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
