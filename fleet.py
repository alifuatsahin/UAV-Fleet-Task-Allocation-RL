from uav import UAV
from mission import Mission
import numpy as np

class Fleet:
    def __init__(self, size: int):
        self._uavs = self._generateUAVs(size)

    def _generateUAVs(self, size):
        return [UAV(i) for i in range(size)]    # maybe give i+1 to match with their code if we get into problems (but should be fine)

    def reset(self):
        [uav.health_initialization() for uav in self._uavs]
        
    def __len__(self):
        return len(self._uavs)
    
    def hasFailed(self) -> bool:
        return any(uav.hasFailed() for uav in self._uavs)
    
    def executeMission(self, hover_distance: float, cruise_distance: float, action: np.ndarray) -> bool:
        hover_uavs = action * hover_distance
        cruise_uavs = action * cruise_distance

        dones = []
        # start individual missions
        for hover, cruise, uav in zip(hover_uavs, cruise_uavs, self._uavs):
            mission = Mission(uav, hover, cruise)
            dones.append(mission.execute())
        
        return any(dones)
    
    def getStats(self) -> np.ndarray:
        return np.concatenate([uav.getStats() for uav in self._uavs])
