from typing import Dict, Any, Set

Entry = Dict[str, Any]
Epoch = int


class History:
    def __init__(self):
        self.history: Dict[Epoch, Entry] = {}
        self.all_keys: Set[str] = set()

    def __getitem__(self, item):
        epochs = [
            epoch for epoch in self.history.keys()
            if item in self.history[epoch].keys()
        ]
        return epochs, [self.history[epoch][item] for epoch in epochs]

    @property
    def all_epochs(self):
        return sorted(self.history.keys())

    def add_entry(self, epoch: Epoch, entry: Entry):
        self.history[epoch] = entry
        self.all_keys.update(entry.keys())

    def by_keys(self):
        return [self[key] for key in self.all_keys]
