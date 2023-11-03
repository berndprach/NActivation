
import os

from datetime import datetime

LOGGING_BASE_FOLDER = os.path.join("outputs", "runs")


class RunLogger:
    def __init__(self, log_folder: str = "other"):
        today = datetime.today()
        self.run_id = today.strftime("%Y%m%d_%H%M%S_%f")

        log_dir = os.path.join(LOGGING_BASE_FOLDER, log_folder, self.run_id)
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir

    def write_to_file(self, filename, contend: str, mode="w"):
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, mode) as f:
            f.write(contend)

    def log_progress(self, progress: str):
        self.write_to_file("progress.txt", progress + "\n", mode="a")
        print(progress)

    def subdir(self, name: str):
        subdir = os.path.join(self.log_dir, name)
        os.makedirs(subdir, exist_ok=True)
        return subdir

    def filepath(self, name: str):
        return os.path.join(self.log_dir, name)

