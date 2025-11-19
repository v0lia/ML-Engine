# timer.py

import time

class Timer:
    def __init__(self, name=None, logger=None):
        self.name = name
        self.logger = logger
        
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        msg = f"[TIMER] Elapsed time: {self.interval:.2f} seconds"
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)
