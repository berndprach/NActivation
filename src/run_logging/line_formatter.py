from typing import Dict, Any


class LineFormatter:
    def __init__(self,
                 column_width: int = 8,
                 seperator: str = " | ",
                 new_line_indent_length: int = 0,
                 ):
        self.column_width = column_width
        self.seperator = seperator
        self.new_line_indent = " "*new_line_indent_length

        self.metric_names = None
        self.header = None

        self.print_header_in = 0

    def create_header(self):
        cw = self.column_width

        train_val_line = []
        for metric_name in self.metric_names:
            if metric_name.startswith("Train"):
                train_val_line.append("Train")
            elif metric_name.startswith("Val"):
                train_val_line.append("Val")
            else:
                train_val_line.append("")
        train_val_entries = [f"{name:{cw}.{cw}}" for name in train_val_line]
        train_val_header = self.seperator.join(train_val_entries)

        print_names = [mn.replace("_", " ").title() + ":"
                       for mn in self.metric_names]
        print_names = [mn.replace("Train ", "") for mn in print_names]
        print_names = [mn.replace("Val ", "") for mn in print_names]

        print_names = [f"{name:{cw}.{cw}}" for name in print_names]
        header = self.seperator.join(print_names)

        return train_val_header + "\n" + header

    def create_line(self, logs: Dict[str, Any]):
        cw = self.column_width

        if self.header is None:
            self.metric_names = sorted(list(logs.keys()))
            self.header = self.create_header()

        line = ""

        if self.print_header_in == 0:
            self.print_header_in = 10
            line = self.header + "\n" + self.new_line_indent + line

        self.print_header_in -= 1

        value_strs = []
        for metric_name in self.metric_names:
            value = logs.get(metric_name, -1.)
            if isinstance(value, float):
                value_str = f"{value:{cw}.4g}"
            else:
                value_str = str(value)
            value_strs.append(f"{value_str:{cw}.{cw}}")

        line += self.seperator.join(value_strs)
        return line
