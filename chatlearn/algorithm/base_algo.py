from abc import ABC, abstractmethod

class BaseAlgorithm(ABC):

    @abstractmethod
    def run(self):
        """
        Run the algorithm.
        """