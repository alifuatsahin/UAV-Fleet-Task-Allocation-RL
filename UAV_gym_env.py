import gym
from gym import spaces
import gym.utils
import numpy as np
from fleet import Fleet
from mission import MissionGenerator

class UAVGymEnv(gym.Env):
    def __init__(self, uav_number, max_distance=100, seed=1234):
        super(UAVGymEnv, self).__init__()
        self._uav_number = uav_number
        self._max_distance = max_distance
        self.Fleet = Fleet(uav_number)
        self._health_state_dim = self.Fleet.getStats().shape[0]
        self.seed(seed)
        self.distance = self.MissionGenerator.generate()
        self._setupActionSpace()
        self._setupObservationSpace()

    def seed(self, seed: int):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.MissionGenerator = MissionGenerator(self._max_distance, self.np_random)

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

    def _getObservation(self) -> np.ndarray:
        state = np.append(self.Fleet.getStats(), self.distance)
        return state

    def _reward(self, done: float) -> float:
        reward = 0
        if not done:
            reward = self.MissionGenerator.current()/100
        # reward -= np.linalg.norm(self._last_health-self.Fleet.getStats()[:-1])
        return reward

    def reset(self):
        self.Fleet.reset()
        info = "reset done"
        return self._getObservation(), info

    def step(self, action: np.ndarray) -> tuple:
        terminate = self.Fleet.executeMission(self.distance, action)
        self.distance = self.MissionGenerator.generate()
        reward = self._reward(terminate)
        truncate = False
        return np.array(self._getObservation()), reward, terminate, truncate, {}