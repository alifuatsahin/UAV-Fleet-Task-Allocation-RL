import gym
from gym import spaces
import gym.utils
import numpy as np

class SimplexGymEnv(gym.Env):
    def __init__(self, simplex_size, seed=1234):
        super(SimplexGymEnv, self).__init__()
        self.simplex_size = simplex_size
        self._generateSimplex()
        self._setupActionSpace()
        self._setupObservationSpace()

    def _setupActionSpace(self):
        self._action_high = 1
        self._action_low = 0
        self.action_space = spaces.Box(low=self._action_low, high=self._action_high, shape=(self.simplex_size,), dtype=np.float32)

    def _setupObservationSpace(self):
        self._obs_low = np.concatenate((np.array([0] * (self.simplex_size-1))))
        
        self._obs_high = np.concatenate((np.array([1] * (self.simplex_size-1)),))

        self.observation_space = spaces.Box(self._obs_low, self._obs_high, dtype=np.float32)

    def _getObservation(self) -> np.ndarray:
        return self.state
    
    def _generateSimplex(self):
        self.state = np.random.dirichlet((1,) * self.simplex_size)
        self.target = np.sort(self.state)
        self.state = np.delete(self.state, np.random.randint(self.simplex_size))

    def _reward(self, action) -> float:
        return np.absolute(np.subtract(self.target, action)).mean()

    def reset(self):
        self._generateSimplex()
        info = {}
        return self._getObservation(), info

    def step(self, action: np.ndarray) -> tuple:
        reward = self._reward(action)
        truncate = False
        terminate = True
        return np.array(self._getObservation()), reward, terminate, truncate, {}