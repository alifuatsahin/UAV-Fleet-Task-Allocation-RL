import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense

class CriticNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dim=256, fc2_dim=256, name='critic', checkpoint_dir = 'tmp\sac'):
        super(CriticNetwork, self).__init__()
        self._fc1_dim = fc1_dim
        self._fc2_dim = fc2_dim
        self._n_actions = n_actions
        self._model_name = name
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_file = os.path.join(self._checkpoint_dir)

        self._fc1 = Dense(self._fc1_dim, activation='relu')
        self._fc2 = Dense(self._fc2_dim, activation='relu')