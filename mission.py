from uav import UAV
import math
import numpy as np
import random

class MissionGenerator:

    def __init__(self, max_distance, seed):
        self._max_distance = max_distance
        self._history = []
        np.random.seed(seed)
        self.generate()
    
    def generate(self):
        distance = np.random.randint(0, self._max_distance)
        prop = np.random.uniform(0.3, 0.7)
        hover_distance = distance*(1-prop)
        cruise_distance = distance*prop
        self._history.append([hover_distance, cruise_distance])
        return hover_distance, cruise_distance

    def current(self):
        return self._history[-1]
    

class Mission:

    def __init__(self, uav: UAV, hover_distance: float, cruise_distance: float):
        self._uav = uav
        self.hover_distance = hover_distance
        self.cruise_distance = cruise_distance
        self._getMissionProfile()
        
    def _getMissionProfile(self):
        self.hover_profile = math.floor(self.hover_distance)
        self.hover_end = self.hover_distance - self.hover_profile

        self.cruise_profile = math.floor(self.cruise_distance)
        self.cruise_end = self.cruise_distance - self.cruise_profile

    def execute(self):
        self._uav.startMission()
        
        for i in range(self.hover_profile):
            self._uav.degrade(1, 0)
        for i in range(self.cruise_profile):
            self._uav.degrade(0, 1)

        self._uav.degrade(self.hover_end.item(), self.cruise_end.item())    
    
        return self._uav.detectFailure()
    
