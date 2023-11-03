from datetime import datetime


class Timer:
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = datetime.now()

    @property
    def seconds_elapsed(self):
        return (datetime.now() - self.start_time).total_seconds()

    # def stop(self):
    #     self.stop_time = datetime.now()

