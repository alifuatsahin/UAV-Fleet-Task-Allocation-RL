import time


class Rate:
    def __init__(self, hz: int):
        self._hz = hz
        self._period = 1/hz
        self._t = time.time()
    
    def sleep(self):
        t = time.time()-self._t
        time.sleep(max(0, self._period - t))
        self._t = t
    
    def reset(self):
        self._t = time.time()
