import numpy as np
from typing import List
import matplotlib.pyplot as plt
from uav import UAVStats
from fleet import Fleet


class Statistics:
    """
    Keeps track of degradation and offers plotting functionalities
    """
    
    # plot strategies for metrics that are alist of multiple values (HOVER_BEARING_HEALTH, HOVER_COIL_HEALTH)
    INDIVIDUAL = 0  # only plotting for a specified index
    AVERAGE = 1     # average over the indices
    LOWEST = 2      # lowest value
    HIGHEST = 3
    
    def __init__(self, fleet: Fleet):
        self._fleet = fleet
        self._degradations: List[np.ndarray] = []   # will be reset when env is reset
        self._total_history_degradations: List[np.ndarray] = []     # won't be reset when env is reset
        self._flown_distances: List[List[float]] = []
        default_val = lambda metric: [0]*4 if metric in UAVStats.MULTIVALUE_METRICS else 0
        self._metric_failures = [{metric: default_val(metric) for metric in UAVStats.METRICS} for _ in range(len(fleet))]
        self.step()     # register initial states
    
    def reset(self):
        self._degradations.clear()
        self._flown_distances.clear()
        
    def step(self):
        """update statistics"""
        degradation = self._fleet.getStats()
        self._degradations.append(degradation)
        self._total_history_degradations.append(degradation)
        self._flown_distances.append([sum(uav.getFlownDistances()) for uav in self._fleet])
        self.detect_failures()
    
    def detect_failures(self):
        for uav_ind, uav in enumerate(self._fleet):
            for metric in UAVStats.METRICS:
                if metric in UAVStats.MULTIVALUE_METRICS:
                    for metric_ind in range(4):
                        if uav.detectComponentFailure(metric, metric_ind):
                            self._metric_failures[uav_ind][metric][metric_ind] += 1
                else:
                    if uav.detectComponentFailure(metric):
                        self._metric_failures[uav_ind][metric] += 1
    
    def get_metric_vals(self, degradations, metric, uav_index, plot_strategy, metric_subindex):
        metric_vals = UAVStats.get_metric(degradations, metric, uav_index)
        if metric in UAVStats.MULTIVALUE_METRICS:
            if plot_strategy == self.INDIVIDUAL: metric_vals = metric_vals[metric_subindex]
            elif plot_strategy == self.AVERAGE: metric_vals = metric_vals.sum(axis=0)
            elif plot_strategy == self.LOWEST: metric_vals = metric_vals.min(axis=0)
            elif plot_strategy == self.HIGHEST: metric_vals = metric_vals.max(axis=0)
            else:
                raise   # TODO
        return metric_vals
    
    def get_metric_label(self, metric: int, plot_strategy: int, metric_subindex: int):
        label = UAVStats.get_metric_name(metric)
        if metric in UAVStats.MULTIVALUE_METRICS:
            if plot_strategy == self.INDIVIDUAL: label %= f" {metric_subindex+1}"
            elif plot_strategy == self.AVERAGE: label %= "s average"
            elif plot_strategy == self.LOWEST: label %= "s lowest"
            elif plot_strategy == self.HIGHEST: label %= "s highest"
        return label

    def plot_one_metric(self, metric: int, uav_index: int = None, plot_strategy: int = None, metric_subindex: int = None, show_legend: bool = False):
        # TODO: implement plot strategies also between UAVs not only between subindices of a multivalue metric
        # TODO: maybe split this function into multiple functions with more specific usages for more simplicity of API
        """
        Plot a specific degradation metric for one or all UAVs across the recorded learning steps.
        
        args
        ----
        metric:
            the specific metric to plot (among UAVStats.METRICS)
        uav_index:
            The uav to plot values for. If it is None, then values for all UAV's will be plotted on the same graph
        plot_strategy:
            Only needs to be specified for multiple value metrics (HOVER_BEARING_HEALTH, HOVER_COIL_HEALTH)
        metric_subindex:
            Only needs to be specified for multiple value metrics when plot strategy is INDIVIDUAL
        """
        if plot_strategy is None:
            plot_strategy = self.LOWEST
            
        degradations = np.stack(self._degradations, axis=-1)
        if uav_index is None and metric in UAVStats.MULTIVALUE_METRICS:
            metric_vals = [self.get_metric_vals(degradations, metric, i, plot_strategy, metric_subindex) for i in range(len(self._fleet))]
        else:
            metric_vals = self.get_metric_vals(degradations, metric, uav_index, plot_strategy, metric_subindex)
            
        step_count = len(self._degradations)
        x = np.arange(step_count)
        plt.xlabel("Number of missions")
        name = self.get_metric_label(metric, plot_strategy, metric_subindex)
        if uav_index is None:
            for i, m in enumerate(metric_vals):
                plt.plot(x, m, label=f"uav {i+1}")
            plt.ylabel(name)
            if show_legend: plt.legend()
        else:
            plt.plot(x, metric_vals)
            plt.ylabel(f"UAV {uav_index+1} {name}")

    def plot_all_metrics(self, uav_index: int, plot_strategy: int = None, metric_subindex: int = None, x_label: str = "Number of missions"):
        """
        Plot all metrics for a specific UAV
        """
        if plot_strategy is None:
            plot_strategy = self.LOWEST
        degradations = np.stack(self._degradations, axis=-1)
        metric_vals = [self.get_metric_vals(degradations, metric, uav_index, plot_strategy, metric_subindex) for metric in UAVStats.METRICS]
        step_count = len(self._degradations)
        x = np.arange(step_count)
        for metric, vals in zip(UAVStats.METRICS, metric_vals):
            label = self.get_metric_label(metric, plot_strategy, metric_subindex)
            plt.plot(x, vals, label=label)
        plt.ylabel(f"Degradations of UAV {uav_index+1}")
        plt.xlabel(x_label)
        plt.legend()

    def plot_flown_distances(self, show_legend: bool = False):
        flown_distances = np.array(self._flown_distances)
        step_count = len(self._degradations)
        x = np.arange(step_count)
        for i in range(len(self._fleet)):
            plt.plot(x, flown_distances[:, i], label=f"uav {i+1}")
        plt.xlabel("Number of missions")
        plt.ylabel(f"Total flown distances")
        if show_legend: plt.legend()

    def plot_failures(self, separate_multivalue_metric: bool = False):
        """plot total of failures accross all UAVs for each component"""
        metric_failure_values = []
        for metric in UAVStats.METRICS:
            if separate_multivalue_metric and metric in UAVStats.MULTIVALUE_METRICS:
                for i in range(4):
                    val = sum(self._metric_failures[uav_index][metric][i] for uav_index in range(len(self._fleet)))
                    metric_failure_values.append(val)
            else:
                count = lambda v: sum(v) if metric in UAVStats.MULTIVALUE_METRICS else v
                val = sum(count(self._metric_failures[uav_index][metric]) for uav_index in range(len(self._fleet)))
                metric_failure_values.append(val)
        metrics = []
        for metric in UAVStats.METRICS:
            name = UAVStats.get_metric_name(metric, health=False)
            if separate_multivalue_metric and metric in UAVStats.MULTIVALUE_METRICS:
                for i in range(4):
                    metrics.append(name % f"{i+1}")
            else:
                if metric in UAVStats.MULTIVALUE_METRICS:
                    name %= "s"
                metrics.append(name)
        plt.bar(metrics, metric_failure_values)
        plt.ylabel("Number of failures")

    def plot_lowest_degredations(self):
        attr_length = 10
        degradations = np.stack(self._degradations, axis=-1)

        for i in range(len(self._fleet)):
            uav_health = degradations[attr_length*i:attr_length*(i+1)]
            lowest_degradations = uav_health.min(axis=0)
            x = np.arange(len(lowest_degradations))
            plt.plot(x, lowest_degradations, label=f"uav {i+1}")
        
        plt.ylabel("Degradations")
        plt.xlabel("Number of missions")
        plt.legend()

    def plot_lowest_healths(self):
        degradations = np.stack(self._degradations, axis=-1)

        lowest_healths = degradations.min(axis=0)
        x = np.arange(len(lowest_healths))
        plt.plot(x, lowest_healths, label=f"min UAV health")
        
        plt.ylabel("Degradations")
        plt.xlabel("Number of missions")
        plt.legend()
