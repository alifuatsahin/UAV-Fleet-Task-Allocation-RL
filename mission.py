from uav import UAV
import math
import numpy as np
import random
import time
from rate import Rate


class MissionGenerator:
    def __init__(self, max_distance, initial_count=1):
        self._max_distance = max_distance
        self._history = []
        for _ in range(initial_count):
            self.generate()
    
    def generate(self):
        distance = random.randint(0, self._max_distance)
        self._history.append(distance)
        return distance

    def current(self):
        return self._history[-1]
    

class Mission:
    HOVER = 1   # TODO: put this in UAV
    CRUISE = 2

    def __init__(self, uav: UAV, distance: float):
        self._uav = uav
        self._distance = distance
        self._mission_profile = self._getMissionProfile()
        self._mission_prep_time = 20

    def _getMissionProfile(self) -> list:
        max_op_time = 50  # in minutes

        approaching_time = math.ceil(self._distance/UAV.CRUISE_SPEED*60)
        reconnaissance_time = max_op_time - (UAV.TAKEOFF_TIME + 2*approaching_time + UAV.LANDING_TIME)
        mission_profile = np.zeros(max_op_time)
        # takeoff and landing approx takes 2 min
        mission_profile[0:UAV.TAKEOFF_TIME] = Mission.HOVER  # takeoff
        mission_profile[-UAV.LANDING_TIME:] = Mission.HOVER  # landing
        # search pattern: mix out of hovering and cruising
        rec_base_time = UAV.TAKEOFF_TIME+approaching_time

        while reconnaissance_time > 0:
            hov_time = random.randint(1, 5)
            mission_profile[rec_base_time:rec_base_time+hov_time] = Mission.HOVER
            cruise_time = random.randint(1, 3)
            mission_profile[rec_base_time+hov_time:rec_base_time+hov_time+cruise_time] = Mission.CRUISE
            reconnaissance_time -= hov_time + cruise_time
            rec_base_time += hov_time + cruise_time

        mission_profile[UAV.TAKEOFF_TIME:UAV.TAKEOFF_TIME + approaching_time] = Mission.CRUISE  # transfer to designated area
        mission_profile[-(UAV.LANDING_TIME + approaching_time):-UAV.LANDING_TIME] = Mission.CRUISE  # transfer from designated area
        return mission_profile

    def execute(self) -> bool:
        """Simuliert den Ablauf der Mission für die übergebene Drohne.

        Args:
            uav(UAV): Die Drohne auf Mission
            distance(int): Distanz der Mission in km
            con(psycopg2.extensions.connection): Verbindung zur Datenbank, in der die Daten der Drohne gespeichert werden

        Returns: None

        """
        self._uav.startMission()

        # increase number of missions (no_missions) by one
        self._uav.number_of_missions += 1
        self._uav.mission_mode = 3  # LORENZ: mission mode = 3 bedeutet in mission preparation
        self._uav.flight_mode = 0  # while in mission preparation UAV is not-flying
        self._uav.mission_progress = 0

        self._uav.rem_mission_len = len(self._mission_profile) + self._mission_prep_time - 1

        # add mission initial status
        rate = Rate(hz=20)
        for _ in range(self._mission_prep_time-1):
            self._uav.rem_mission_len = self._uav.rem_mission_len - 1
            rate.sleep()

        # start mission
        self._uav.mission_mode = 1 # SAR mission TODO: get rid of mission_mode

        rate.reset()
        for i, mission_step in enumerate(len(self._mission_profile)):
            self._uav.degradation(mission_step)  # new health values for single step
            self._uav.mission_progress = (i + 1) / len(self._mission_profile) * 100  # 100 - (1 - (len(mission[:i + 1]) / len(mission))) * 100
            self._uav.rem_mission_len = self._uav.rem_mission_len - 1
            if self._uav.flight_mode == 1:
                self._uav.hbmx_count = self._uav.hbmx_count + 1
                self._uav.hcmx_count = self._uav.hcmx_count + 1
            if self._uav.flight_mode == 2:
                self._uav.pbmx_count = self._uav.pbmx_count + 1
                self._uav.pcmx_count = self._uav.pcmx_count + 1

            # DnP part
            if self._uav.detectFailure():
                self._uav.stopMission()
                return False

            rate.sleep()

        self._uav.stopMission()
        return True
