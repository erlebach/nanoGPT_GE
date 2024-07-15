import time
import threading
from contextlib import contextmanager
from collections import defaultdict

class PerfCounterTimer:
    timings = defaultdict(list)

    def __init__(self, name=""):
        self.name = name

    @contextmanager
    def timeit(self):
        start_time = time.perf_counter()
        yield
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        PerfCounterTimer.timings[self.name].append(elapsed_time)

    def __call__(self):
        return self.timeit()

    @classmethod
    def report(cls, msg=""):
        if msg is not "":
            print(f"\n{msg}")
        for name, times in cls.timings.items():
            total_time = sum(times)
            count = len(times)
            print(f"Name: {name}, Count: {count}, Total Time: {total_time:.6f} seconds, Timings: {total_time/count} each")


if __name__ == '__main__':
    # Usage example
    timer = PerfCounterTimer()

    # Simulating multiple timed blocks
    with PerfCounterTimer("gordon").timeit():
        time.sleep(0.5)
    with PerfCounterTimer("gordon").timeit():
        time.sleep(0.5)

    with PerfCounterTimer("frances").timeit():
        time.sleep(0.7)
    with PerfCounterTimer("frances").timeit():
        time.sleep(0.7)

    with PerfCounterTimer("ggordon").timeit():
        time.sleep(0.3)
    with PerfCounterTimer("ggordon").timeit():
        time.sleep(0.3)

    # Print the report
    PerfCounterTimer.report()
