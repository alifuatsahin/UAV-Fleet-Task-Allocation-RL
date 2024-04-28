import gym
from gym import spaces
import numpy as np
from uav import UAV
from fleet import Fleet
from mission import SearchDistanceGenerator


class UAVGymEnv(gym.Env):
    def __init__(self, uav_number):
        super(UAVGymEnv, self).__init__()
        self._fleet = Fleet(size=uav_number)
        param_count = 11
        self._health_state_dim = param_count * len(self._fleet)
        self._max_distance = 100
        self._setupActionSpace()
        self._setupObservationSpace()
        self._mission_generator = SearchDistanceGenerator(self._max_distance)

    def _setupActionSpace(self):
        self._action_lim = np.array([1] * len(self._fleet))
        self._action_low = np.zeros_like(self._action_lim)
        self.action_space = spaces.Box(self._action_low, self._action_lim, dtype=np.float32)

    def _setupObservationSpace(self):
        self._obs_low = np.concatenate((np.array([0] * self._health_state_dim),
                                        np.array([0])))
        self._obs_high = np.concatenate((np.array([1] * self._health_state_dim),
                                        np.array([self._max_distance])))

        self.observation_space = spaces.Box(self._obs_low, self._obs_high, dtype=np.float32)

    def _getObservation(self) -> np.ndarray:
        return self._fleet.getStats()

    def _reward(self):
        return 0
    
    def _checkHealth(self):
        return False

    def reset(self):
        self._fleet.reset()

    def step(self, action: np.ndarray):
        total_search_distance = self._mission_generator.current()
        success = self._fleet.executeMission(total_search_distance, action)
        reward = self._reward()
        terminated = not success
        truncated = False
        self._mission_generator.generate()
        return np.array(self._getObservation()), reward, terminated, truncated, {}
