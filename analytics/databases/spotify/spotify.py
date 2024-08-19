from abc import ABC, abstractmethod
# ~/miniconda3/envs/dev/bin/dmypy


class SomeClass(ABC):
    @abstractmethod
    def root(self, value):
        self._root = value
