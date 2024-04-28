import gym
from gym import spaces
import numpy as np

class UAVGymEnv(gym.Env):
    def __init__(self, uav_number):
        super(UAVGymEnv, self).__init__()
        self._uav_number = uav_number
        self._health_state_dim = 11 * self._uav_number
        self._max_distance = 100
        self._setupActionSpace()
        self._setupObservationSpace()

    def _setupActionSpace(self):
        self._action_lim = np.array([1] * self._uav_number)
        self._action_low = np.zeros_like(self._action_lim)
        self.action_space = spaces.Box(self._action_low, self._action_lim, dtype=np.float32)

    def _setupObservationSpace(self):
        self._obs_low = np.concatenate((np.array([0] * self._health_state_dim),
                                        np.array([0])))
        self._obs_high = np.concatenate((np.array([1] * self._health_state_dim),
                                        np.array([self._max_distance])))

        self.observation_space = spaces.Box(self._obs_low, self._obs_high, dtype=np.float32)

    def _getObservation(self):
        return 0

    def _reward(self):
        return 0
    
    def _checkHealth(self):
        return False

    def reset(self):
        # self._fleet = UAVfleet(self._uav_number)
        pass

    def step(self, action):
        self._fleet.ApplyAction(action)
        reward = self._reward()
        done = True
        return np.array(self._getObservation()), reward, done, {}