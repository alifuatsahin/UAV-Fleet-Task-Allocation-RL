import numpy as np
from typing import List
import matplotlib.pyplot as plt
from uav import UAVStats


class Statistics:
    """
    Keeps track of degradation and offers plotting functionalities
    """
    
    # plot strategies for metrics that are alist of multiple values (HOVER_BEARING_HEALTH, HOVER_COIL_HEALTH)
    INDIVIDUAL = 0  # only plotting for a specified index
    AVERAGE = 1     # average over the indices
    LOWEST = 2      # lowest value
    HIGHEST = 3
    
    def __init__(self):
        self._degradations: List[np.ndarray] = []   # will be reset when env is reset
        self._total_history_degradations: List[np.ndarray] = []     # won't be reset when env is reset
    
    def reset(self):
        self._degradations.clear()
        
    def step(self, degradation: np.ndarray):
        self._degradations.append(degradation)
        self._total_history_degradations.append(degradation)
    
    def plot_degradation(self, metric: int, uav_index: int = None, plot_strategy: int = None, metric_subindex: int = None):
        # TODO: implement plot strategies also between UAVs not only between subindices of a multivalue metric
        # TODO: maybe split this function into multiple functions with more specific usages for more simplicity of API
        """
        Plot a specific degradation metric for one or all UAVs across the recorded learning steps.
        
        args
        ----
        metric:
            the specific metric to plot (among UAVStats.METRICS)
        uav_index:
            The uav to plot values for. If it is None, then values for all UAV's will be plotted on the same graph (only supported for single value metrics for now)
        plot_strategy:
            Only needs to be specified for multiple value metrics (HOVER_BEARING_HEALTH, HOVER_COIL_HEALTH)
        metric_subindex:
            Only needs to be specified for multiple value metrics when plot strategy is INDIVIDUAL
        """
        
        if uav_index is None and metric in UAVStats.MULTIVALUE_METRICS:
            raise NotImplementedError("plotting values for all UAVs on the same graph not implemented yet for multivalue metrics, please specify a uav index")
        degradations = np.stack(self._degradations, axis=-1)
        metric_vals = UAVStats.get_metric(degradations, metric, uav_index)
        if metric in UAVStats.MULTIVALUE_METRICS:
            if plot_strategy == self.INDIVIDUAL: metric_vals = metric_vals[metric_subindex]
            elif plot_strategy == self.AVERAGE: metric_vals = metric_vals.sum(axis=0)
            elif plot_strategy == self.LOWEST: metric_vals = metric_vals.min(axis=0)
            elif plot_strategy == self.HIGHEST: metric_vals = metric_vals.max(axis=0)
            else:
                raise   # TODO
            
        step_count = len(self._degradations)
        x = np.arange(step_count)
        plt.xlabel("Time (in minutes I think)")
        name = UAVStats.get_metric_name(metric)
        if metric in UAVStats.MULTIVALUE_METRICS:
            if plot_strategy == self.INDIVIDUAL: name %= f" {metric_subindex+1}"
            elif plot_strategy == self.AVERAGE: name %= "s average"
            elif plot_strategy == self.LOWEST: name %= "s lowest"
            elif plot_strategy == self.HIGHEST: name %= "s highest"
        if uav_index is None:
            for i, m in enumerate(metric_vals):
                plt.plot(x, m, label=f"uav {i+1}")
            plt.ylabel(name)
            plt.legend()
        else:
            plt.plot(x, metric_vals)
            plt.ylabel(f"UAV {uav_index+1} {name}")
