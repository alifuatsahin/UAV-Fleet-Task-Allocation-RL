import numpy as np
import time
import random


def mx_procedure(uav_instance,
                 con,
                 cur,
                 time_scale):

    # service times
    battery_swap = 30
    battery_service = 40
    hover_bearing_mx = 60
    hover_coil_mx = 120
    pusher_bearing_mx = 80
    pusher_coil_mx = 150
    mx_th = 0.2

    # setting end mission UAV parameters
    uav_instance.mission_progress = 100  # remaining mission length
    uav_instance.mission_mode = 5  # change status: maintenance
    uav_instance.flight_mode = 0

    components_health = np.zeros((10))
    components_health[0:4] = uav_instance.hover_bearing_health
    components_health[4:8] = uav_instance.hover_coil_health
    components_health[8] = uav_instance.pusher_bearing_health
    components_health[9] = uav_instance.pusher_coil_health
    # components_health[0] = 0.12

    # print("!!!!!!!!!!!!!!!! UAV {} health before mx: ".format(uav_instance.uav_id), components_health, uav_instance.battery_level)
    # print(len(np.where(components_health <= mx_th)[0]) > 0)
    # check for failed components
    if len(np.where(components_health <= mx_th)[0]) > 0:
        # print("UAV {} in maintenance".format(uav_instance.uav_id))
        failed_component_id = np.where(components_health < mx_th)[0]+1
        # for i in range(len(failed_component_id)):
        #     if failed_component_id <= 4:
        #         # print("Bearing failure at hover motor ", failed_component_id)
        #     if 5 <= failed_component_id <= 8:
        #         # print("Coil failure at hover motor ", failed_component_id-4)
        #     if failed_component_id == 9:
        #         # print("Bearing failure at pusher motor", failed_component_id)
        #     if failed_component_id == 10:
                # print("Coil failure at pusher motor", failed_component_id)

        # check for faulty components
        failure_thresholds = np.zeros((10))
        failure_thresholds[0:4] = uav_instance.hover_bearing_failure_appearance
        failure_thresholds[4:8] = uav_instance.hover_coil_failure_appearance
        failure_thresholds[8] = uav_instance.pusher_bearing_failure_appearance
        failure_thresholds[9] = uav_instance.pusher_coil_failure_appearance

        # print("Components health: ", components_health)
        # print("Failure thresholds: ", failure_thresholds)

        faulty_component_id = np.where(components_health < failure_thresholds)[0]+1
        # print("Faulty ids: ", faulty_component_id)
        mx_times = []
        for i in range(len(faulty_component_id)):
            if faulty_component_id[i] <= 4:
                # print("Faulty bearing at hover motor ", faulty_component_id[i])
                mx_times.append(hover_bearing_mx)
            if 5 <= faulty_component_id[i] <= 8:
                # print("Faulty coil at hover motor ", faulty_component_id[i]-4)
                mx_times.append(hover_coil_mx)
            if faulty_component_id[i] == 9:
                # print("Faulty bearing at pusher motor", faulty_component_id[i])
                mx_times.append(pusher_bearing_mx)
            if faulty_component_id[i] == 10:
                # print("Faulty coil at pusher motor", faulty_component_id[i])
                mx_times.append(pusher_coil_mx)

        # mx procedure
        start_time = time.time()

        # swapping battery
        uav_instance.battery_level = 1 - uav_instance.battery_loading_cycles * 0.0003
        uav_instance.battery_loading_cycles = uav_instance.battery_loading_cycles + 1
        # in case of degraded battery, do battery service
        if uav_instance.battery_level < 0.7:
            # print("UAV {} in battery service".format(uav_instance.uav_id))
            uav_instance.battery_loading_cycles = 0
            uav_instance.battery_level = 1

        for i in range(len(faulty_component_id)):
            if faulty_component_id[i] <= 4:
                # print("Before mx: ", uav_instance.hover_bearing_health[faulty_component_id[i]-1])
                uav_instance.hover_bearing_health[faulty_component_id[i]-1] = random.uniform(0.95, 1.0)
                # print("After mx: ", uav_instance.hover_bearing_health[faulty_component_id[i]-1])
                uav_instance.hover_bearing_factors[faulty_component_id[i]-1] = 1.0
            if 5 <= faulty_component_id[i] <= 8:
                # print("Before mx: ", uav_instance.hover_coil_health[faulty_component_id[i]-5])
                uav_instance.hover_coil_health[faulty_component_id[i]-5] = random.uniform(0.95, 1.0)
                # print("After mx: ", uav_instance.hover_coil_health[faulty_component_id[i]-5])
                uav_instance.hover_coil_factors[faulty_component_id[i]-5] = 1.0
            if faulty_component_id[i] == 9:
                # print("Before mx: ", uav_instance.pusher_bearing_health)
                uav_instance.pusher_bearing_health = random.uniform(0.95, 1.0)
                # print("After mx: ", uav_instance.pusher_bearing_health)
                uav_instance.pusher_bearing_factor = 1.0
            if faulty_component_id[i] == 10:
                # print("Before mx: ", uav_instance.pusher_coil_health)
                uav_instance.pusher_coil_health = random.uniform(0.95, 1.0)
                # print("After mx: ", uav_instance.pusher_coil_health)
                uav_instance.pusher_coil_factor = 1.0
        elapsed_time = time.time() - start_time
        if elapsed_time < time_scale:
            time.sleep(time_scale - elapsed_time)
            uav_instance.store_to_database(con, cur)

        for _ in range(max(mx_times)):
            start_time = time.time()
            elapsed_time = time.time() - start_time
            if elapsed_time < time_scale:
                time.sleep(time_scale - elapsed_time)
            uav_instance.store_to_database(con, cur)

        start_time = time.time()
        uav_instance.mission_progress = 0  # reset mission progression bar

        uav_instance.mission_mode = 2  # change status: available
        uav_instance.flight_mode = 0  # not-flying

        elapsed_time = time.time() - start_time
        if elapsed_time < time_scale:
            time.sleep(time_scale - elapsed_time)
        uav_instance.store_to_database(con, cur)

    # battery mx after normal mission ending without detected failures and faults
    else:
        # print("UAV {} in battery service".format(uav_instance.uav_id))
        for q in range(battery_swap):
            start_time = time.time()
            # print("UAV {} - {}/{} of battery swap done".format(uav.uav_id, q + 1, battery_swap))
            elapsed_time = time.time() - start_time
            if elapsed_time < time_scale:
                time.sleep(time_scale - elapsed_time)
            uav_instance.store_to_database(con, cur)
        # load battery before mission while on ground, respect max cap degradation
        uav_instance.battery_level = 1 - uav_instance.battery_loading_cycles * 0.0003
        uav_instance.battery_loading_cycles = uav_instance.battery_loading_cycles + 1

        if uav_instance.battery_level < 0.7:
            # print("UAV {} in battery service")
            for q in range(battery_service):
                uav_instance.battery_loading_cycles = 0
                start_time = time.time()
                # print("UAV {} - {}/{} of battery service done".format(uav.uav_id, q + 1, battery_service))
                elapsed_time = time.time() - start_time
                if elapsed_time < time_scale:
                    time.sleep(time_scale - elapsed_time)

                uav_instance.store_to_database(con, cur)
            uav_instance.battery_level = 1

        start_time = time.time()
        uav_instance.mission_progress = 0  # reset mission progression bar

        uav_instance.mission_mode = 2  # change status: available
        uav_instance.flight_mode = 0  # not-flying

        elapsed_time = time.time() - start_time
        if elapsed_time < time_scale:
            time.sleep(time_scale - elapsed_time)
        uav_instance.store_to_database(con, cur)
