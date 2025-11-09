import time
from contextlib import contextmanager


class SleepTimer:
    def __init__(self, frequency):
        """Timer to help respect a given frequency

        :param frequency: frequency to respect
        """
        self.frequency = frequency
        
    def start_timer(self, frequency=None):
        """Start the timer"""
        self.start_time = time.time()
        if frequency is not None:
            self.frequency = frequency

    def sleep(self):
        """Sleep the correct amount of time to respect the frequency"""
        elapsed_time = time.time() - self.start_time
        sleep_time = self.update_sleep_time(self.frequency, elapsed_time)
        time.sleep(sleep_time)
        return sleep_time

    def update_sleep_time(self, frequency, elapsed_time):
        """Compute the time we need to sleep to respect the update frequency

        :param frequency: update state frequency
        :param elapsed_time: time already used to compute the state
        :return: time needed to sleep in addition to elapsed time to respect the frequency
        """
        # if we use the freq, compute the correct sleep time
        if float(frequency) > 0.0:
            perfect_time = 1.0 / float(frequency)
            sleep_time = max(perfect_time - elapsed_time, 0)
        # Else set it to zero
        else:
            sleep_time = 0
        return sleep_time


@contextmanager
def sleep_timer(timer=None, freq=None):
    """Context manager to time a block of code and sleep to respect a given frequency"""
    if timer is None and freq is None:
        raise ValueError("If 'timer' is not provided, 'freq' must be specified.")
    if timer is None:
        timer = SleepTimer(freq)
    timer.start_timer(freq)
    try:
        yield
    finally:
        timer.sleep()