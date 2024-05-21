from uav import UAV
from mission import Mission
import numpy as np

class Fleet:
    def __init__(self, size: int, seed: int):
        self._uavs = self._generateUAVs(size, seed)

    def _generateUAVs(self, size, seed):
        return [UAV(i, seed) for i in range(size)]    # maybe give i+1 to match with their code if we get into problems (but should be fine)

    def reset(self):
        [uav.health_initialization() for uav in self._uavs]
        
    def __len__(self):
        return len(self._uavs)
    
    def __iter__(self):
        for uav in self._uavs:
            yield uav
    
    def detectFailure(self) -> bool:
        return any(uav.detectFailure() for uav in self._uavs)
    
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

    def getMostUsedUAV(self) -> int:
        """get the uav that flew the most"""
        return max(range(len(self._uavs)), key=lambda i: sum(self._uavs[i].getFlownDistances()))
