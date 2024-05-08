import gym
from gym import spaces
import numpy as np
from fleet import Fleet
from mission import MissionGenerator
from stats import Statistics


class UAVGymEnv(gym.Env):
    def __init__(self, uav_number, max_distance=100):
        super(UAVGymEnv, self).__init__()
        self._uav_number = uav_number
        self._max_distance = max_distance
        self.Fleet = Fleet(uav_number)
        self.MissionGenerator = MissionGenerator(self._max_distance)
        self._health_state_dim = self.Fleet.getStats().shape[0]
        self._setupActionSpace()
        self._setupObservationSpace()
        self._last_health = self.Fleet.getStats()[:-1]
        self._statistics = Statistics()

    def _setupActionSpace(self):
        self._action_lim = np.array([1] * len(self.Fleet))
        self._action_low = np.zeros_like(self._action_lim)
        self.action_space = spaces.Box(self._action_low, self._action_lim, dtype=np.float32)

    def _setupObservationSpace(self):
        self._obs_low = np.concatenate((np.array([0] * self._health_state_dim),
                                        np.array([0])))
        self._obs_high = np.concatenate((np.array([1] * self._health_state_dim),
                                        np.array([self._max_distance])))

        self.observation_space = spaces.Box(self._obs_low, self._obs_high, dtype=np.float32)

    def _getObservation(self):
        distance = self.MissionGenerator.current()
        state = np.append(self.Fleet.getStats(), distance)
        return np.array([state])

    def _reward(self, done):
        reward = 0
        if not done:
            reward = self.MissionGenerator.current()
        # reward -= np.linalg.norm(self._last_health-self.Fleet.getStats()[:-1])
        return reward

    def reset(self):
        self.Fleet.reset()
        self._statistics.reset()
        return self._getObservation()

    def step(self, action):
        self._statistics.step(self.Fleet.getStats())
        self._last_health = self.Fleet.getStats()[:-1]
        distance = self.MissionGenerator.generate()
        done = self.Fleet.executeMission(distance, action)
        reward = self._reward(done)
        return np.array(self._getObservation()), reward, done, {}

    def plot_degradation(self, metric: int, uav_index: int = None, plot_strategy: int = None, metric_subindex: int = None):
        """
        Example usages:
            # a single value metric such as PUSHER_BEARING_HEALTH can be plotted for all UAVs on the same graph (not supported yet for multivalue metrics)
            >>> env.plot_degradation(UAVStats.PUSHER_BEARING_HEALTH, uav_index=None)
            
            # multivalue metric HOVER_BEARING_HEALTH plotted for uav index 0 (the lowest health among the 4 hover bearings plotted at each step)
            >>> env.plot_degradation(UAVStats.HOVER_BEARING_HEALTH, uav_index=0, plot_strategy=Statistics.LOWEST)
            
            # multivalue metric HOVER_BEARING_HEALTH plotted for uav index 1 (health of bearing 0 plotted at each step)
            >>> env.plot_degradation(UAVStats.HOVER_BEARING_HEALTH, uav_index=1, plot_strategy=Statistics.INDIVIDUAL, metric_subindex=0)
            
            
        For more details, see Statistics.plot_degradation.
        """
        self._statistics.plot_degradation(metric, uav_index, plot_strategy, metric_subindex, fleet_length=self._uav_number)

