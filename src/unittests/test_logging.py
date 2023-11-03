
import unittest

from src import run_logging


class TestUtils(unittest.TestCase):
    def test_line_formatter(self):
        formatter = run_logging.LineFormatter(column_width=4)
        logs = {"Aaa": 1., "Bbbbbbbb": 2., "C": 0.03}
        lines = formatter.create_line(logs)
        print(lines)
        header = lines.split("\n")[0]
        values = lines.split("\n")[1]
        self.assertEqual(header, "Aaa: | Bbbb | C:  ")
        self.assertEqual(values, "   1 |    2 | 0.03")

