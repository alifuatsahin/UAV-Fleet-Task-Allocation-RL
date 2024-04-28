import numpy as np
from uav import UAV
import math
import random

# AV performance data
max_op_time = 50  # in minutes

# two loadcases:
# hovermode = 1
# cruisemode = 2

# generating mission profiles


def delivery_mission(distance: int) -> np.ndarray:
    """Delivery missions vary in length so it is possible to give a distance in km as an input.
    Mission profile is then constructed out of fixed takeoff and landing phase in hovering mode and in between energy
    efficient cruising mode

    Args:
        distance(int): Die Distanz der gesamten Mission.

    Returns:
        np.ndarray: Einen Array, dessen Index i den Flugmodus zur i-ten Minute beschreibt.

    """
    mission_time = math.ceil(distance/UAV.CRUISE_SPEED*60) + 4  # in min, rounded up
    mission_profile = np.zeros(mission_time)
    # takeoff and landing approx takes 2 min
    mission_profile[0:UAV.TAKEOFF_TIME] = 1  # takeoff
    mission_profile[-UAV.LANDING_TIME:] = 1  # landing
    # cruising/delivery phase
    mission_profile[UAV.TAKEOFF_TIME:-UAV.LANDING_TIME] = 2  # cruising
    return mission_profile


def reconnaissance_mission(distance: int) -> np.ndarray:
    """Reconnaissance mission is defined through fixed takeoff and landing phase as well as a distance to approach
        to designated area.

    Args:
        distance(int): Die Distanz zum Missionsziel

    Returns:
        np.ndarray: Einen Array, dessen Index i den Flugmodus zur i-ten Minute beschreibt.

     """
    approaching_time = math.ceil(distance/UAV.CRUISE_SPEED*60)
    reconnaissance_time = max_op_time - (UAV.TAKEOFF_TIME + 2*approaching_time + UAV.LANDING_TIME)
    mission_profile = np.zeros(max_op_time)
    # takeoff and landing approx takes 2 min
    mission_profile[0:UAV.TAKEOFF_TIME] = 1  # takeoff
    mission_profile[-UAV.LANDING_TIME:] = 1  # landing
    # search pattern: mix out of hovering and cruising
    rec_base_time = UAV.TAKEOFF_TIME+approaching_time

    while reconnaissance_time > 0:
        hov_time = random.randint(1, 5)
        mission_profile[rec_base_time:rec_base_time+hov_time] = 1
        cruise_time = random.randint(1, 3)
        mission_profile[rec_base_time+hov_time:rec_base_time+hov_time+cruise_time] = 2
        reconnaissance_time -= hov_time + cruise_time
        rec_base_time += hov_time + cruise_time

    mission_profile[UAV.TAKEOFF_TIME:UAV.TAKEOFF_TIME + approaching_time] = 2  # transfer to designated area
    mission_profile[-(UAV.LANDING_TIME + approaching_time):-UAV.LANDING_TIME] = 2  # transfer from designated area
    return mission_profile
