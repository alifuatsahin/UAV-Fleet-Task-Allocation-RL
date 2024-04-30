import gym
from gym import spaces
import numpy as np
from fleet import Fleet
from mission import MissionGenerator

class UAVGymEnv(gym.Env):
    def __init__(self, uav_number ,max_distance=100):
        super(UAVGymEnv, self).__init__()
        self._uav_number = uav_number
        self.Fleet = Fleet(uav_number)
        self.MissionGenerator = MissionGenerator(max_distance)
        self._health_state_dim = Fleet.getStats().shape[0]
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
        distance = MissionGenerator.current()
        state = np.concatenate(self.Fleet.getStats(), np.array([distance]))
        return state

    def _reward(self, done):
        return 1 if done else 0

    def reset(self):
        Fleet.reset()

    def step(self, action):
        distance = MissionGenerator.generate()
        done = self.Fleet.executeMission(distance, action)
        reward = self._reward()
        return np.array(self._getObservation()), reward, done, {}