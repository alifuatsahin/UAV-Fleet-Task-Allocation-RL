from __future__ import annotations
import numpy as np
import random
import math
from typing import List


pusher_bearing_factor = [1]
pusher_coil_factor = [1]

class UAV:
    """Die Klasse UAV modelliert eine hybride Drohne mit einem Pusher-Motor, 8 Hover-Motoren und einer Batterie."""
    CRUISE_SPEED = 60  # in kmh
    TAKEOFF_TIME = 2  # in minutes
    LANDING_TIME = 2  # in minutes

    def __init__(self, uav_id: int):
        # self.hbmx_count = 0
        # self.hcmx_count = 0
        # self.pbmx_count = 0
        # self.pcmx_count = 0
        self.uav_id = uav_id
        self.flight_mode = None
        self.hover_bearing_health = np.zeros(4)
        self.hover_bearing_factors = np.zeros(4)
        self.hover_coil_health = np.zeros(4)
        self.hover_coil_factors = np.zeros(4)
        self.pusher_bearing_health = 0.0
        self.pusher_bearing_factor = 1
        self.pusher_coil_health = 0.0
        self.pusher_coil_factor = 1
        self.battery_level = 1.0
        self.battery_loading_cycles = 0
        self.hover_bearing_failure_appearance = np.empty(4)
        self.hover_coil_failure_appearance = np.empty(4)
        self.pusher_bearing_failure_appearance = None
        self.pusher_coil_failure_appearance = None
        self.number_of_missions = 0
        self.mission_mode = 0  # --> 0: on delivery mission, 1: on SAR mission, 2: available, 3: R2H, 4: in maintenance
        self.mission_id = 0
        # self.mission_progress = None
        self.rem_mission_len = 0
        self.health_initialization()
        self.failure_detection = False  #"False"
        # self.critic_comp = "Not yet detected!"
        # self.rul = None
        # self.contingency = "False"
        # self.asset_risk = "False"
        # self.mission_risk = "False"

        self._in_mission = False
        self._flown_distances: List[float] = []     # distances flown for each mission

    def health_initialization(self) -> None:
        """Initialsiert die Gesundheitswerte der Motoren, deren failure-Appearance und die Ladung der Batterie.

        Args: None

        Returns: None

        """
        self.flight_mode = 0  # set UAV not-flying
        self.mission_mode = 2  # set UAV available
        # self.mission_progress = 0

        # initial hover health
        for i in range(len(self.hover_bearing_health)):
            self.hover_bearing_health[i] = random.uniform(0.80, 1.0)
        for i in range(len(self.hover_bearing_factors)):
            self.hover_bearing_factors[i] = 1.0
        for i in range(len(self.hover_bearing_failure_appearance)):
            self.hover_bearing_failure_appearance[i] = round(random.uniform(0.5, 0.55), 3)

        for i in range(len(self.hover_coil_health)):
            self.hover_coil_health[i] = random.uniform(0.90, 1.0)
        for i in range(len(self.hover_coil_factors)):
            self.hover_coil_factors[i] = 1.0
        for i in range(len(self.hover_coil_failure_appearance)):
            self.hover_coil_failure_appearance[i] = round(random.uniform(0.80, 0.85), 3)

        # initial pusher health
        self.pusher_bearing_health = random.uniform(0.80, 1.0)
        self.pusher_bearing_failure_appearance = round(random.uniform(0.45, 0.5), 3)
        self.pusher_bearing_factor = 1.0

        self.pusher_coil_health = random.uniform(0.90, 1.0)
        self.pusher_coil_failure_appearance = round(random.uniform(0.80, 0.85), 3)
        self.pusher_coil_factor = 1.0

        # initial battery health
        self.battery_level = 1
        
        self._flown_distances = []

    def health_index(self) -> int:  # 26
        """Berechnet das Minimum der Gesundheitswerte der Motoren.

        Args: None

        Returns: None

        """
        return min([*self.hover_bearing_health, *self.hover_coil_health,
                    self.pusher_bearing_health, self.pusher_coil_health])

    def degradation(self, mission_mode: int) -> None:
        """Simuliert den Verschleiß der Motoren und die Entladung der Batterie abhängig vom aktuellen Missions-Modus.

        Args:
            mission_mode(int): Der aktuelle Missionsmodus -> 0=no-flying, 1=hover, 2=cruise

        Returns: None

        """
        self.flight_mode = mission_mode

        if mission_mode == 1:

            """Hover bearing degradation:"""
            hover_bearing_deg_values = np.empty(len(self.hover_bearing_health))
            for j in range(len(self.hover_bearing_health)):
                if self.hover_bearing_health[j] > self.hover_bearing_failure_appearance[j]:
                    hover_bearing_deg_values[j] = self.hover_bearing_factors[j]*round(random.uniform(1, 3) * 0.0001, 6)
                    self.hover_bearing_factors[j] *= 1.001
                else:
                    hover_bearing_deg_values[j] = self.hover_bearing_factors[j]*round((random.uniform(1, 3) * 0.0001), 6)
                    self.hover_bearing_factors[j] *= 1.006

            self.hover_bearing_health -= hover_bearing_deg_values

            """Hover coil degradation:"""
            hover_coil_deg_values = np.empty(len(self.hover_coil_health))
            for j in range(len(self.hover_coil_health)):
                if self.hover_coil_health[j] > self.hover_coil_failure_appearance[j]:
                    hover_coil_deg_values[j] = self.hover_coil_factors[j]*round(random.uniform(0.00005, 0.0001), 5)
                    self.hover_coil_factors[j] *= 1.001
                else:
                    hover_coil_deg_values[j] = self.hover_coil_factors[j]*round(random.uniform(0.00005, 0.0001), 5)
                    self.hover_coil_factors[j] *= 1.004

            self.hover_coil_health -= hover_coil_deg_values

            """Battery discharge: way higher discharge rate more rapid changes in discharge due to hover mode"""
            hover_health_bat_fac = 1+(1-(min(min(self.hover_bearing_health), min(self.hover_coil_health))))*0.33
            discharge_rate = round((random.uniform(0.03, 0.07)*hover_health_bat_fac), 2)
            self.battery_level -= discharge_rate

        if mission_mode == 2:
            """Pusher bearing degradation:"""
            if self.pusher_bearing_health > self.pusher_bearing_failure_appearance:
                pusher_bearing_deg_rate = self.pusher_bearing_factor*round(random.uniform(1, 3) * 0.000023, 6)
                self.pusher_bearing_factor *= 1.001
            else:
                pusher_bearing_deg_rate = self.pusher_bearing_factor*round((random.uniform(1, 3) * 0.000023), 6)
                self.pusher_bearing_factor *= 1.004

            self.pusher_bearing_health -= pusher_bearing_deg_rate

            """Pusher coil degradation:"""
            if self.pusher_coil_health > self.pusher_coil_failure_appearance:
                pusher_coil_deg_rate = self.pusher_coil_factor*round(random.uniform(1, 3) * 0.000005, 6)
                self.pusher_coil_factor *= 1.001
            else:
                pusher_coil_deg_rate = self.pusher_coil_factor*round(random.uniform(1, 3) * 0.000005, 6)
                self.pusher_coil_factor *= 1.002

            self.pusher_coil_health -= pusher_coil_deg_rate

            """Battery discharge: lower discharge rate, more constant discharge due to cruise mode"""
            pusher_health_bat_fac = 1+((1-min(self.pusher_bearing_health, self.pusher_coil_health))*0.33)
            discharge_rate = round((random.uniform(0.003, 0.01)*pusher_health_bat_fac), 3)
            self.battery_level -= discharge_rate
        
    def detectFailure(self):
        component_healths = []
        component_healths.extend(list(self.hover_bearing_health))
        component_healths.extend(list(self.hover_coil_health))
        component_healths.append(float(self.pusher_bearing_health))
        component_healths.append(float(self.pusher_coil_health))
        component_healths = np.round(np.asarray(component_healths), 2)

        # component_failure_ths = []
        # component_failure_ths.extend(list(self.hover_bearing_failure_appearance))
        # component_failure_ths.extend(list(self.hover_coil_failure_appearance))
        # component_failure_ths.append(float(self.pusher_bearing_failure_appearance))
        # component_failure_ths.append(float(self.pusher_coil_failure_appearance))
        # component_failure_ths = np.round(np.asarray(component_failure_ths), 2)

        # health_difs = np.subtract(component_healths, component_failure_ths)
        #self.failure_detection = '{}'.format(any(health_difs < 0))
        self.failure_detection = any(component_healths < 0)
        return self.failure_detection
    
    def startMission(self):    # TODO: coordinate this with other redundant attributes
        self._flown_distances.append(0)
        self._in_mission = True
    
    def stopMission(self):     # TODO: coordinate this with other redundant attributes
        self._in_mission = False

    def inMission(self) -> bool:
        return self._in_mission
    
    def hasFailed(self) -> bool:
        return self.failure_detection

    def flyMinute(self):
        self._flown_distances[-1] += self.CRUISE_SPEED / 60
    
    def getFlownDistances(self):
        return self._flown_distances
        
    def getStats(self) -> np.ndarray:
        return UAVStats(self).get()


class UAVStats:
    """
    Class for gathering stats encoding and decoding, to make it easier if we want to change those mechanics later 
    (for instance by adding back battery levels to the stats, which they are currently not).
    """
    
    HOVER_BEARING_HEALTH = 0
    HOVER_COIL_HEALTH = 1
    PUSHER_BEARING_HEALTH = 2
    PUSHER_COIL_HEALTH = 3
    BATTERY_LEVEL = 4
    
    METRICS = [HOVER_BEARING_HEALTH, HOVER_COIL_HEALTH, PUSHER_BEARING_HEALTH, PUSHER_COIL_HEALTH]  #, BATTERY_LEVEL]
    MULTIVALUE_METRICS = [HOVER_BEARING_HEALTH, HOVER_COIL_HEALTH]
    
    STAT_NAMES = {
        HOVER_BEARING_HEALTH: "hover bearing%s health",
        HOVER_COIL_HEALTH: "hover coil%s health",
        PUSHER_BEARING_HEALTH: "pusher bearing health",
        PUSHER_COIL_HEALTH: "pusher coil health",
        BATTERY_LEVEL: "battery level"
    }
    
    def __init__(self, uav: UAV):
        self._uav = uav
    
    def get(self):
        """get stats encoded as a numpy array"""
        return np.concatenate((
            self._uav.hover_bearing_health,
            self._uav.hover_coil_health,
            np.array([self._uav.pusher_bearing_health]),
            np.array([self._uav.pusher_coil_health]),
            #np.array([self._uav.battery_level])
        ))
    
    @staticmethod
    def get_metric(stats: np.ndarray, metric: int, uav_index: int = None) -> np.ndarray:
        """
        Get a metric of the stats that have been encoded to numpy array.
        args
        ----
        stats:
            a numpy array of shape (<nb metrics>*<nb uavs>) x <nb of steps recorded>
        metric:
            The metric to isolate among UAVStats.METRICS

        returns the value of this metric for the given uav_index across all the steps
        """
        attr_lengths = [4, 4, 1, 1]     # [4, 4, 1, 1, 1] with battery level
        total_length = sum(attr_lengths)
        if uav_index is None:   # TODO: for now only works for attributes of length 1
            start = sum(attr_lengths[:metric])
            return stats[start::total_length]
        start = total_length*uav_index + sum(attr_lengths[:metric])
        length = attr_lengths[metric]
        return stats[start] if length == 1 else stats[start: start+length]

    
    @classmethod
    def get_metric_name(cls, metric: int):
        name = cls.STAT_NAMES[metric]
        return name
