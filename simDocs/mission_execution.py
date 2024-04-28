from mission_generator import delivery_mission, reconnaissance_mission
import numpy as np
from uav import UAV
import time
import pandas as pd
import random
import psycopg2
#from mx_predetermined import mx_procedure as mp
from mx_at_fault import mx_procedure as mp
import matplotlib.pyplot as plt

time_scale = 0.05  # time taken for one minute in simulation (in seconds)
mission_prep_time = 20


def failure_detection(uav):
    component_healths = []
    component_healths.extend(list(uav.hover_bearing_health))
    component_healths.extend(list(uav.hover_coil_health))
    component_healths.append(float(uav.pusher_bearing_health))
    component_healths.append(float(uav.pusher_coil_health))
    component_healths = np.round(np.asarray(component_healths), 2)

    component_failure_ths = []
    component_failure_ths.extend(list(uav.hover_bearing_failure_appearance))
    component_failure_ths.extend(list(uav.hover_coil_failure_appearance))
    component_failure_ths.append(float(uav.pusher_bearing_failure_appearance))
    component_failure_ths.append(float(uav.pusher_coil_failure_appearance))
    component_failure_ths = np.round(np.asarray(component_failure_ths), 2)

    health_difs = np.subtract(component_healths, component_failure_ths)
    uav.failure_detection = '{}'.format(any(health_difs < 0))


def mission_execution(uav: UAV, distance: int, con):
    """Simuliert den Ablauf der Mission für die übergebene Drohne.

    Args:
        uav(UAV): Die Drohne auf Mission
        distance(int): Distanz der Mission in km
        con(psycopg2.extensions.connection): Verbindung zur Datenbank, in der die Daten der Drohne gespeichert werden

    Returns: None

    """
    # cursor
    cur = con.cursor()

    # increase number of missions (no_missions) by one
    uav.number_of_missions += 1
    uav.mission_mode = 3  # LORENZ: mission mode = 3 bedeutet in mission preparation
    uav.flight_mode = 0  # while in mission preparation UAV is not-flying
    uav.mission_progress = 0

    mission = delivery_mission(distance) if uav.mission_type == 0 else reconnaissance_mission(distance)
    uav.rem_mission_len = len(mission) + mission_prep_time - 1

    # print("UAV {} mission prep".format(uav.uav_id))
    # add mission initial status
    for r in range(mission_prep_time-1):
        start_time = time.time()
        elapsed_time = time.time() - start_time
        if elapsed_time < time_scale:
            time.sleep(time_scale - elapsed_time)
        uav.rem_mission_len = uav.rem_mission_len - 1
        uav.store_to_database(con, cur)
        # print("{}/{} Mission prep time".format(r + 1, mission_prep_time))

    # start mission
    if uav.mission_type == 0:
        uav.mission_mode = 0  # set mission_mode to delivery mission

    if uav.mission_type == 1:
        uav.mission_mode = 1  # set mission_mode to SAR mission


    for i in range(len(mission)):
        start_time = time.time()

        uav.degradation(mission[i])  # new health values for single step
        uav.mission_progress = (i + 1) / len(mission) * 100  # 100 - (1 - (len(mission[:i + 1]) / len(mission))) * 100
        uav.rem_mission_len = uav.rem_mission_len - 1
        if uav.flight_mode == 1:
            uav.hbmx_count = uav.hbmx_count + 1
            uav.hcmx_count = uav.hcmx_count + 1
        if uav.flight_mode == 2:
            uav.pbmx_count = uav.pbmx_count + 1
            uav.pcmx_count = uav.pcmx_count + 1

        # DnP part
        failure_detection(uav=uav)

        # print("{}/{} mission done".format(i + 1, len(mission)))
        elapsed_time = time.time() - start_time
        if elapsed_time < time_scale:
            time.sleep(time_scale - elapsed_time)
        uav.store_to_database(con, cur)


        if 0.1 < uav.battery_level < 0.3 and len(mission) - i > distance + UAV.LANDING_TIME:
            # print("UAV {} safe R2H triggered".format(uav.uav_id))
            uav.mission_mode = 4

            r2h_distance = int(int(distance) + int(distance) // 10)  # int(float(distance + int(distance * 0.1)))
            r2h_mission = np.zeros(r2h_distance + int(UAV.LANDING_TIME))
            r2h_mission[0:r2h_distance] = 2
            r2h_mission[-UAV.LANDING_TIME:] = 1
            uav.rem_mission_len = len(r2h_mission)
            # print('Safe R2H triggered with a duration of ', len(r2h_mission), 'mins')

            for j in range(len(r2h_mission)):
                start_time = time.time()

                uav.degradation(r2h_mission[j])  # new health values for single step
                uav.mission_progress = (j + 1) / len(r2h_mission) * 100
                if uav.flight_mode == 1:
                    uav.hbmx_count = uav.hbmx_count + 1
                    uav.hcmx_count = uav.hcmx_count + 1
                if uav.flight_mode == 2:
                    uav.pbmx_count = uav.pbmx_count + 1
                    uav.pcmx_count = uav.pcmx_count + 1

                # print("{}/{} R2H mission done".format(j + 1, len(r2h_mission)))
                elapsed_time = time.time() - start_time
                if elapsed_time < time_scale:
                    time.sleep(time_scale - elapsed_time)
                uav.store_to_database(con, cur)



                uav.rem_mission_len = uav.rem_mission_len - 1

                # DnP part
                failure_detection(uav=uav)

            # print('!!!NOTICE: UAV {} proceeds to EARLY mission ending'.format(uav.uav_id))

            mp(uav_instance=uav,
               con=con,
               cur=cur,
               time_scale=time_scale
               )

            # close cursor
            cur.close()
            break

    if i+1 == len(mission) and uav.mission_mode == 0 or uav.mission_mode == 1:
        # print('!!!NOTICE: UAV {} proceeds to NORMAL mission ending'.format(uav.uav_id))

        mp(uav_instance=uav,
           con=con,
           cur=cur,
           time_scale=time_scale
           )

        # close cursor
        cur.close()
