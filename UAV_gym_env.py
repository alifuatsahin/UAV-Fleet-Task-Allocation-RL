import gym
from gym import spaces
import gym.utils
import numpy as np
from fleet import Fleet
from mission import MissionGenerator
from stats import Statistics


class UAVGymEnv(gym.Env):
    def __init__(self, uav_number, max_distance=100, seed=1234):
        super(UAVGymEnv, self).__init__()
        self._uav_number = uav_number
        self._max_distance = max_distance
        self.Fleet = Fleet(uav_number, seed)
        self._health_state_dim = self.Fleet.getStats().shape[0]
        self.MissionGenerator = MissionGenerator(self._max_distance, seed)
        self.hover_distance, self.cruise_distance = self.MissionGenerator.generate()
        self._setupActionSpace()
        self._setupObservationSpace()
        self._statistics = Statistics(self.Fleet)

    def _setupActionSpace(self):
        self._action_high = 1
        self._action_low = 0
        self.action_space = spaces.Box(low=self._action_low, high=self._action_high, shape=(self._uav_number,), dtype=np.float32)

    def _setupObservationSpace(self):
        self._obs_low = np.concatenate((np.array([0] * self._health_state_dim),
                                        np.array([0, 0])))
        self._obs_high = np.concatenate((np.array([1] * self._health_state_dim),
                                        np.array([self._max_distance, self._max_distance])))

        self.observation_space = spaces.Box(self._obs_low, self._obs_high, dtype=np.float32)

    def _getObservation(self) -> np.ndarray:
        state = np.append(self.Fleet.getStats(), self.hover_distance)
        state = np.append(state, self.cruise_distance)
        return state

    def _reward(self, done: float) -> float:
        reward = 0
        if not done:
            reward = sum(self.MissionGenerator.current())
        # reward -= np.linalg.norm(self._last_health-self.Fleet.getStats()[:-1])
        return reward

    def reset(self):
        self.Fleet.reset()
        self._statistics.reset()
        info = "reset done"
        return self._getObservation(), info

    def step(self, action: np.ndarray) -> tuple:
        terminate = self.Fleet.executeMission(self.hover_distance, self.cruise_distance, action)
        self._statistics.step()
        reward = self._reward(terminate)
        self.hover_distance, self.cruise_distance = self.MissionGenerator.generate()
        truncate = False
        return np.array(self._getObservation()), reward, terminate, truncate, {}

    def plot_one_metric(self, metric: int, uav_index: int = None, plot_strategy: int = None, metric_subindex: int = None, show_legend: bool = False):
        """
        Plot a certain metric for one or all UAVs
        Example usages:
            # a single value metric such as PUSHER_BEARING_HEALTH can be plotted for all UAVs on the same graph
            >>> env.plot_one_metric(UAVStats.PUSHER_BEARING_HEALTH, uav_index=None)
            
            # multivalue metric HOVER_BEARING_HEALTH plotted for uav index 0 (the lowest health among the 4 hover bearings plotted at each step)
            >>> env.plot_one_metric(UAVStats.HOVER_BEARING_HEALTH, uav_index=0, plot_strategy=Statistics.LOWEST)
            
            # multivalue metric HOVER_BEARING_HEALTH plotted for uav index 1 (health of bearing 0 plotted at each step)
            >>> env.plot_one_metric(UAVStats.HOVER_BEARING_HEALTH, uav_index=1, plot_strategy=Statistics.INDIVIDUAL, metric_subindex=0)
            
            
        For more details, see Statistics.plot_one_metric.
        """
        self._statistics.plot_one_metric(metric, uav_index, plot_strategy, metric_subindex, show_legend=show_legend)

    def plot_all_metrics(self, uav_index: int, plot_strategy: int = None, metric_subindex: int = None):
        """
        Plot all metrics for a single UAV
        see Statistic.plot_all_metrics
        """
        self._statistics.plot_all_metrics(uav_index, plot_strategy, metric_subindex)

    def plot_flown_distances(self, show_legend: bool = False):
        self._statistics.plot_flown_distances(show_legend=show_legend)

    def plot_lowest_degredations(self):
        self._statistics.plot_lowest_degredations()
