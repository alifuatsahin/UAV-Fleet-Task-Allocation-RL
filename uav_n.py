import numpy as np
import random

class UAV:

    def __init__(self, uav_id: int):
        self.uav_id = uav_id
        self.p_hover_dist = 0
        self.hover_dist = 0
        self.p_cruise_dist = 0
        self.cruise_dist = 0

        self.hover_bearing_health = np.zeros(4)
        self.hover_bearing_factors = np.zeros(4)
        self.hover_coil_health = np.zeros(4)
        self.hover_coil_factors = np.zeros(4)

        self.pusher_bearing_health = 0.0
        self.pusher_bearing_factor = 1
        self.pusher_coil_health = 0.0
        self.pusher_coil_factor = 1

        self.hover_bearing_failure_appearance = np.empty(4)
        self.hover_coil_failure_appearance = np.empty(4)

        self.pusher_bearing_failure_appearance = None
        self.pusher_coil_failure_appearance = None

        self.health_initialization()
        self.failure_detection = False

    def health_initialization(self) -> None:
        self.flight_mode = 0
        
        for i in range(len(self.hover_bearing_health)):
            self.hover_bearing_health[i] = random.uniform(0.80, 1.0)
            self.hover_bearing_factors[i] = 1.0
            self.hover_bearing_failure_appearance[i] = round(random.uniform(0.5, 0.55), 3)

            self.hover_coil_health[i] = random.uniform(0.80, 1.0)
            self.hover_coil_factors[i] = 1.0
            self.hover_coil_failure_appearance[i] = round(random.uniform(0.5, 0.55), 3)

        self.pusher_bearing_health = random.uniform(0.80, 1.0)
        self.pusher_bearing_factor = 1
        self.pusher_bearing_failure_appearance = round(random.uniform(0.5, 0.55), 3)

    def degrade(self, hover: float, cruise: float) -> None:

        if  hover > 0:
            hover_bearing_deg_values = np.empty(len(self.hover_bearing_health))
            for j in range(len(self.hover_bearing_health)):
                if self.hover_bearing_health[j] > self.hover_bearing_failure_appearance[j]:
                    hover_bearing_deg_values[j] = self.hover_bearing_factors[j]*round(random.uniform(1, 3) * 0.0001, 6)
                    self.hover_bearing_factors[j] *= 1.001
                else:
                    hover_bearing_deg_values[j] = self.hover_bearing_factors[j]*round((random.uniform(1, 3) * 0.0001), 6)
                    self.hover_bearing_factors[j] *= 1.006

            self.hover_bearing_health -= hover*hover_bearing_deg_values

            hover_coil_deg_values = np.empty(len(self.hover_coil_health))
            for j in range(len(self.hover_coil_health)):
                if self.hover_coil_health[j] > self.hover_coil_failure_appearance[j]:
                    hover_coil_deg_values[j] = self.hover_coil_factors[j]*round(random.uniform(0.00005, 0.0001), 5)
                    self.hover_coil_factors[j] *= 1.001
                else:
                    hover_coil_deg_values[j] = self.hover_coil_factors[j]*round(random.uniform(0.00005, 0.0001), 5)
                    self.hover_coil_factors[j] *= 1.006
                    
            self.hover_coil_health -= hover*hover_coil_deg_values

        if cruise > 0:
            if self.pusher_bearing_health > self.pusher_bearing_failure_appearance:
                pusher_bearing_deg_rate = self.pusher_bearing_factor*round(random.uniform(1, 3) * 0.00005, 6)
                self.pusher_bearing_factor *= 1.001
            else:
                pusher_bearing_deg_rate = self.pusher_bearing_factor*round((random.uniform(1, 3) * 0.00005), 6)
                self.pusher_bearing_factor *= 1.006

            self.pusher_bearing_health -= cruise*pusher_bearing_deg_rate

            if self.pusher_coil_health > self.pusher_coil_failure_appearance:
                pusher_coil_deg_rate = self.pusher_coil_factor*round(random.uniform(0.00005, 0.0001), 5)
                self.pusher_coil_factor *= 1.001
            else:
                pusher_coil_deg_rate = self.pusher_coil_factor*round(random.uniform(0.00005, 0.0001), 5)
                self.pusher_coil_factor *= 1.006

            self.pusher_coil_health -= cruise*pusher_coil_deg_rate

    def detectFailure(self) -> bool:
        component_healths = []
        component_healths.extend(list(self.hover_bearing_health))
        component_healths.extend(list(self.hover_coil_health))
        component_healths.append(float(self.pusher_bearing_health))
        component_healths.append(float(self.pusher_coil_health))

        return any(component_healths < 0)

    def getStats(self) -> np.ndarray:
        return np.concatenate((
            self.hover_bearing_health,
            self.hover_coil_health,
            np.array([self.pusher_bearing_health]),
            np.array([self.pusher_coil_health]))
        )
