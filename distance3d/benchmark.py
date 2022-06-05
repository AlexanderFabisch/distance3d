"""Benchmark utilities."""
import time


class Timer:
    """Collect timing information from benchmarks."""
    def __init__(self):
        self._start_time = {}
        self.total_time_ = {}

    def start(self, name):
        self._start_time[name] = time.time()

    def stop(self, name):
        stop_time = time.time()
        assert name in self._start_time
        return stop_time - self._start_time[name]

    def stop_and_add_to_total(self, name):
        duration = self.stop(name)
        current_total = self.total_time_.get(name, 0.0)
        self.total_time_[name] = current_total + duration
