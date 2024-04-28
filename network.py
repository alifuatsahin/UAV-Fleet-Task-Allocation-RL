import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
import tensorflow_probability as tfp

class CriticNetwork(keras.Model):
    def __init__(self, n_actions, nn1_dim=256, nn2_dim=256, name='critic', checkpoint_dir = 'tmp\sac'):
        super(CriticNetwork, self).__init__()
        self._n_actions = n_actions
        self._model_name = name
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_file = os.path.join(self._checkpoint_dir, name +'\sac')

        self._network = keras.Sequential([
            Dense(nn1_dim, activation='relu'),
            Dense(nn2_dim, activation='relu'),
            Dense(1, activation=None)
        ])

    def call(self, state, action):
        q = self._network(tf.concat(action, state), axis=1)

        return q
    
class ValueNetwork(keras.Model):
    def __init__(self, nn1_dim=256, nn2_dim=256, name='value', checkpoint_dir = 'tmp\sac'):
        super(ValueNetwork, self).__init__()
        self._model_name = name
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_file = os.path.join(self._checkpoint_dir, name +'_sac')

        self._network = keras.Sequential([
            Dense(nn1_dim, activation='relu'),
            Dense(nn2_dim, activation='relu'),
            Dense(1, activation=None)
        ])

    def call(self, state):
        v = self._network(state)

        return v
    
class ActorNetwork(keras.Model):
    def __init__(self, n_actions, nn1_dim=256, nn2_dim=256, name='actor', checkpoint_dir = 'tmp\sac'):
        super(ActorNetwork, self).__init__()
        self._n_actions = n_actions
        self._model_name = name
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_file = os.path.join(self._checkpoint_dir, name +'\sac')
        self._noise = 1e-16

        self._network = keras.Sequential([
            Dense(nn1_dim, activation='relu'),
            Dense(nn2_dim, activation='relu'),
            Dense(self._n_actions, activation=None)
        ])

    def call(self, state):
        alpha = self._network(state)

        return alpha
    
    def sample_dirichlet(self, state):
        alpha = self.call(state)
        prob = tfp.distributions.Dirichlet(alpha)

        action = prob.sample()
        log_probs = prob.log_prob(action)
        log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)

        return action, log_probs

