import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self._m_size = max_size
        self._m_count = 0
        self._state_m = np.zeros((self._m_size, input_shape))
        self._new_state_m = np.zeros((self._m_size, input_shape))
        self._action_m = np.zeros((self._m_size, n_actions))
        self._reward_m = np.zeros((self._m_size))
        self._terminal_m = np.zeros(self._m_size, dtype=np.bool)

    def store(self, state, action, reward, state_, done):
        id = self._m_count % self._m_size

        self._state_m[id] = state
        self._new_state_m[id] = state_
        self._action_m[id] = action
        self._reward_m[id] = reward
        self._terminal_m[id] = done

        self._m_count += 1

    def sample(self, batch_size):
        max_m = min(self._m_count, self._m_size)

        batch = np.random.choice(max_m, batch_size)

        states = self._state_m[batch]
        states_ = self._new_state_m[batch]
        actions = self._action_m[batch]
        rewards = self._reward_m[batch]
        dones = self._terminal_m[batch]

        return states, states_, actions, rewards, dones