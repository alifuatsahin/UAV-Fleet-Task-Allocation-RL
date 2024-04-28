import threading
from mission_type import MissionType
from uav import UAV
from pathlib import Path
from db import init_database
import numpy as np
from mission_execution import mission_execution
import time
import multiprocessing as mp
import psycopg2.extensions
from degradation_validity import deg_plot
from uav_health_monitor import health_plot
from usage_monitoring import usage_plot
import random
# from test_lstm_rest_api import rt_rul_assessment


len_uav_array = 27
time_scale = 0.05
NUMBER_OF_UAVS = 10
# no_ex_missions = 5000  # Anzahl Missionen pro Drohne


def main(show_graphs=True):
    """Die main-Funktion des Projekts:
    Zuerst wird eine Verbindung zur Datenbank hergestellt, dann wird in einer Schleife
    für jede Drone Missionen simuliert und anschließend ausgewählte Werte in Graphen dargestellt."""

    # erstellt data Ordner, falls nicht vorhanden
    Path("data").mkdir(parents=True, exist_ok=True)

    con = init_database(NUMBER_OF_UAVS)
    # cursor
    cur = con.cursor()

    Path("data").mkdir(parents=True, exist_ok=True)

    uavs = []
    for i in range(NUMBER_OF_UAVS):
        uav = UAV(i+1)
        uavs.append(uav)

        # insert initial health into database
        uav.store_to_database(con, cur)
        print("UAV {} initialized".format(i+1))
        
    if not show_graphs:
        return uavs, con


def mission_management(uavs, con: psycopg2.extensions.connection):
    cur = con.cursor()
    mission_queue = np.genfromtxt("mission_queue.csv", delimiter=",")
    pool = mp.Pool(NUMBER_OF_UAVS)

    mission_no = 1

    fleet_availability = np.zeros((NUMBER_OF_UAVS))
    active_missions = []
    # print("\nTotal number of missions: ", len(mission_queue))
    while True:
        print('\n', mission_no, ' / ', len(np.unique(mission_queue[:, 0])))
        if mission_no == len(np.unique(mission_queue[:, 0])):
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Mission execution stopped!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            break
        # check pending mission and get number of UAV needed
        uav_needed = len(mission_queue[mission_queue[:, 0] == mission_no])

        # check number of UAV available
        for i in range(10):
            cur.execute('select mission_mode from uav{} order by index desc limit 1'.format(i + 1))
            uav_status = np.array(cur.fetchall())
            fleet_availability[i] = int(uav_status)

            # cur.execute("select mission_id from uav{} order by index desc limit 1".format(i + 1))
            current_mission_id = np.array(cur.fetchall())
            if current_mission_id != 0:
                active_missions.append(int(current_mission_id))
        active_missions = list(set(active_missions))
        available_mission_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for i in range(len(active_missions)):
            available_mission_ids.remove(active_missions[i])

        # print("Before mission fleet availability: ", fleet_availability)
        # print("Active Mission IDs: ", active_missions)
        # print("Available Mission IDs: ", available_mission_ids)
        uav_available = len(np.where(fleet_availability == 2)[0])
        print(fleet_availability)
        print('Available UAV: ', uav_available, ' -  Needed UAV: ', uav_needed)

        # assign mission
        if uav_needed <= uav_available:

            if mission_queue[mission_queue[:, 0] == mission_no][0, 1] == 0:
                # delivery mission
                print("Delivery mission in queue")
                # get available UAV [integration of health dependend selection HERE; instead of min]
                available_uav_id = random.choice(np.where(fleet_availability == 2)[0]) + 1
                # get latest health values and update class
                cur.execute("select hb1, hb2, hb3, hb4, hb1_fac, hb2_fac, hb3_fac, hb4_fac, hc1, hc2, hc3, hc4, "
                                "hc1_fac, hc2_fac, hc3_fac, hc4_fac, psb, psb_fac, psc, psc_fac, bat_h, no_missions "
                                "from uav{} order by index desc limit 1".format(available_uav_id))
                latest_uav_values = np.array(cur.fetchall())[0]
                uavs[available_uav_id-1].mission_mode = 3
                uavs[available_uav_id-1].hover_bearing_health = latest_uav_values[0:4]
                uavs[available_uav_id-1].hover_bearing_factors = latest_uav_values[4:8]
                uavs[available_uav_id-1].hover_coil_health = latest_uav_values[8:12]
                uavs[available_uav_id-1].hover_coil_factors = latest_uav_values[12:16]
                uavs[available_uav_id-1].pusher_bearing_health = latest_uav_values[16]
                uavs[available_uav_id-1].pusher_bearing_factor = latest_uav_values[17]
                uavs[available_uav_id-1].pusher_coil_health = latest_uav_values[18]
                uavs[available_uav_id-1].pusher_coil_factor = latest_uav_values[19]
                uavs[available_uav_id-1].battery_level = latest_uav_values[20]
                uavs[available_uav_id-1].number_of_missions = latest_uav_values[21]
                uavs[available_uav_id-1].mission_id = available_mission_ids[0]
                available_mission_ids.pop(0)

                #define mission parameters
                mission_type = list(MissionType)[0]
                uavs[available_uav_id-1].mission_type = mission_type.value
                mission_len = mission_queue[mission_queue[:, 0] == mission_no][0, 3]
                print("UAV to be sent on delivery mission: ", available_uav_id)
                fleet_availability[available_uav_id - 1] = False
                # pool.apply_async(mission_execution, [uavs[available_uav_id-1], mission_len, con])
                threading.Thread(target=mission_execution, args=(uavs[available_uav_id-1], mission_len, con)).start()
                # threading.Thread(target=rt_rul_assessment, args=(uavs[available_uav_id - 1], con)).start()

                uavs[available_uav_id-1].mission_mode = 3

            if mission_queue[mission_queue[:, 0] == mission_no][0, 1] == 1:
                # reconnaissance mission
                print("SAR mission in queue")
                # get available UAV [integration of health dependend selection HERE]
                mission_len = mission_queue[mission_queue[:, 0] == mission_no][0, 3]
                for j in range(uav_needed):
                    available_uav_id = random.choice(np.where(fleet_availability == 2)[0]) + 1
                    # get latest health values and update class
                    cur.execute("select hb1, hb2, hb3, hb4, hb1_fac, hb2_fac, hb3_fac, hb4_fac, hc1, hc2, hc3, hc4, "
                                    "hc1_fac, hc2_fac, hc3_fac, hc4_fac, psb, psb_fac, psc, psc_fac, bat_h, no_missions "
                                    "from uav{} order by index desc limit 1".format(available_uav_id))
                    latest_uav_values = np.array(cur.fetchall())[0]
                    uavs[available_uav_id-1].mission_mode = 3
                    uavs[available_uav_id-1].hover_bearing_health = latest_uav_values[0:4]
                    uavs[available_uav_id-1].hover_bearing_factors = latest_uav_values[4:8]
                    uavs[available_uav_id-1].hover_coil_health = latest_uav_values[8:12]
                    uavs[available_uav_id-1].hover_coil_factors = latest_uav_values[12:16]
                    uavs[available_uav_id-1].pusher_bearing_health = latest_uav_values[16]
                    uavs[available_uav_id-1].pusher_bearing_factor = latest_uav_values[17]
                    uavs[available_uav_id-1].pusher_coil_health = latest_uav_values[18]
                    uavs[available_uav_id-1].pusher_coil_factor = latest_uav_values[19]
                    uavs[available_uav_id-1].battery_level = latest_uav_values[20]
                    uavs[available_uav_id-1].number_of_missions = latest_uav_values[21]
                    uavs[available_uav_id-1].mission_id = available_mission_ids[0]

                    # define mission parameters
                    mission_type = list(MissionType)[1]
                    uavs[available_uav_id-1].mission_type = mission_type.value
                    print("UAV to be sent on SAR mission: ", available_uav_id)
                    fleet_availability[available_uav_id - 1] = False

                    # pool.apply_async(mission_execution, [uavs[available_uav_id-1], mission_len, con])
                    threading.Thread(target=mission_execution, args=(uavs[available_uav_id - 1], mission_len, con)).start()
                    # threading.Thread(target=rt_rul_assessment, args=(uavs[available_uav_id - 1], con)).start()

                available_mission_ids.pop(0)

            mission_no += 1

        active_missions = []  # empty mission ids list to update in next check round
        time.sleep(5*time_scale)

    # close cursor
    cur.close()
    # # close database
    # con.close()


if __name__ == '__main__':

    init_uavs = main(False)
    mission_management(init_uavs[0], init_uavs[1])
    health_plot()
    usage_plot()
    deg_plot()
