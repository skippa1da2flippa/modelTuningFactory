from abc import ABC, abstractmethod
import pandas as pd


class BaseFactory(ABC):

    @abstractmethod
    def tuningProcess(self, dataset: pd.DataFrame):
        pass

    @abstractmethod
    def modelsBuilder(self):
        pass


