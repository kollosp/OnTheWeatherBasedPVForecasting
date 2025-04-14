import time

class ExecutionTimer:
    def __init__(self):
        self.start = 0
        self.start_delta = 0
        self.end = 0

    @property
    def seconds_elapsed(self):
        """Count how many seconds was measured by 'with' clause (timer stopped)"""
        return self.end - self.start

    @property
    def seconds(self):
        """Count how many seconds already elapsed (timer not stopped)"""
        return time.time() - self.start

    @property
    def seconds_delta(self):
        """Count how many seconds elapsed from last 'seconds_delta' execution (timer not stopped)"""
        tmp = time.time()
        delta = tmp - self.start_delta
        self.start_delta = time.time()
        return delta

    def __enter__(self):
        self.start = time.time()
        self.start_delta = self.start
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        # Exception handling here
        self.end = time.time()