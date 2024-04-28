from uav import UAV
from mission import Mission
import numpy as np
import threading
from rate import Rate


class Fleet:
    def __init__(self, size: int):
        self._uavs = self._generateUAVs(size)

    def _generateUAVs(self, size):
        return [UAV(i) for i in range(size)]    # maybe give i+1 to match with their code if we get into problems (but should be fine)

    def reset(self):
        self._uavs = self._generateUAVs(len(self._uavs))
        
    def __len__(self):
        return len(self._uavs)
    
    def hasFailed(self) -> bool:
        return any(uav.hasFailed() for uav in self._uavs)
    
    def executeMission(self, total_distance: float, action: np.ndarray) -> bool:
        """return True for succes, False for failure"""
        search_distances = action * total_distance

        # start individual missions
        for distance, uav in zip(search_distances, self._uavs):
            mission = Mission(uav, distance)
            thread = threading.Thread(target=mission.execute)
            thread.start()
        
        # wait for execution
        rate = Rate(hz=20)
        while True:
            if not any(uav.inMission() for uav in self._uavs):
                return not self.hasFailed()
            rate.sleep()    # release ressources for other threads
    
    def getStats(self) -> np.ndarray:
        return np.concatenate([uav.getStats() for uav in self._uavs])
