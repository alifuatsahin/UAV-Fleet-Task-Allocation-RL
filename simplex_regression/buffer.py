import random
import numpy as np
import os
import pickle

class ReplayBuffer:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))

        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.memory)
    
    def save(self, filename, env_name, suffix='', save_dir=None):
        if not os.path.exists("checkpoints/"):
            os.makedirs("checkpoints/")

        if save_dir is None:
            save_dir = "checkpoints/buffer_{}_{}".format(env_name, suffix)
        print("... saving replay buffer checkpoint ...")

        with open(save_dir, 'wb') as f:
            pickle.dump(f, self.memory)

    def load(self, load_dir):
        print("... loading replay buffer checkpoint ...")

        with open(load_dir, 'rb') as f:
            self.memory = pickle.load(f)
            self.position  = len(self.memory) % self.capacity


