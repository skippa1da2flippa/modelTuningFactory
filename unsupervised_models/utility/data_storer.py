from typing import TypeVar
import joblib

T = TypeVar('T')


class DataHandler:
    @staticmethod
    def storeData(data: T, path: str):
        joblib.dump(data, path)

    @staticmethod
    def getData(path: str) -> T:
        return joblib.load(path)
