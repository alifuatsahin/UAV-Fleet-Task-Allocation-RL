from uav_n import UAV
import math
import numpy as np
import random

class MissionGenerator:
    def __init__(self, max_distance, np_random=None):
        self._max_distance = max_distance
        self._history = []
        self.np_random = np_random
        self.generate()
    
    def generate(self):
        # distance = self.np_random.integers(0, self._max_distance)
        distance = self._max_distance
        self._history.append(distance)
        return distance

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
        self.hover_end = self.hover_profile - self.hover_profile

        self.cruise_profile = math.floor(self.cruise_distance)
        self.cruise_end = self.cruise_profile - self.cruise_profile

    def execute(self):
        for i in range(self.hover_profile):
            self._uav.degrade(1, 0)
        for i in range(self.cruise_profile):
            self._uav.degrade(0, 1)

        self._uav.degrade(self.hover_end, self.cruise_end)

        return self._uav.detectFailure()
    
