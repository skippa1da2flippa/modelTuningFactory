import time
import numpy as np
import pandas as pd
from numpy import ndarray, array, append
from sklearn.base import BaseEstimator
from unsupervised_models.base_classes.base_factory import BaseFactory
from sklearn.metrics import rand_score
from unsupervised_models.PCA_reduction.PCA import PcaFeatsWrapper


class BaseModelFactory(BaseFactory):

    def __init__(
            self, models: list, candidates: list[float], X: pd.DataFrame, y: ndarray
    ):
        self._pcaHandler: PcaFeatsWrapper = PcaFeatsWrapper()
        self._models: list = models
        self._candidates: list[float] = candidates
        self._X: pd.DataFrame = X
        self._y: ndarray = y
        self._outcomeCollection: ndarray[dict[str, float]] = np.array([])

    def tuningProcess(self, dataset: pd.DataFrame, returnBestModel: bool = False) -> tuple:

        # at the end of the cycle below this array shall be populated with the models R.I
        randomIndexes: ndarray[float] = array([])

        predictions: ndarray[ndarray[int]] = array([])

        times: ndarray[float] = array([])

        for model in self._models:
            start = time.time()

            # train the model and see how the model it's doing with the training data
            prediction = model.fit_predict(dataset)

            end = time.time()

            predictions = array([prediction]) if not predictions.size else append(predictions, [prediction], axis=0)

            times = append(times, end - start)

            # compare the model prediction with the actual prediction and store it in a collection
            randomIndexes = np.append(randomIndexes, rand_score(self._y, prediction))

        # find the index related to the max values
        bestIdx = np.argmax(randomIndexes)

        # find the right hyper-parameter
        hyperParam: float = self._candidates[bestIdx]

        bestTime: float = times[bestIdx]

        # find the best model
        bestModel: BaseEstimator = self._models[bestIdx]

        if not returnBestModel:
            return hyperParam, predictions, randomIndexes, bestTime
        else:
            return hyperParam, predictions, randomIndexes, bestTime, bestModel

    def modelsBuilder(self) -> ndarray[dict]:
        # retrieve the data after the dimensionality changes (PCA)
        dataFrameCollection: list[pd.DataFrame] = self._pcaProcess()

        for idx, dataFrame in enumerate(dataFrameCollection):
            # for each dimensionality reduction a tuning process is instantiated
            tempTup: tuple[float, ndarray[ndarray[int]], ndarray[float], float, BaseEstimator] = \
                self.tuningProcess(dataFrame, True)

            # append the result of the tuning process
            self._outcomeCollection = np.append(self._outcomeCollection, {
                "dimensions": idx + 2,
                "hyper-param": tempTup[0],
                "predictions": tempTup[1],
                "rand-index": np.max(tempTup[2]),
                "time": tempTup[3],
                "model": tempTup[4]
            })

            print(f"PCA: {idx + 1}, remaining: {9 - idx}")

        return self._outcomeCollection

    def _pcaProcess(self) -> list[pd.DataFrame]:
        return self._pcaHandler.pcaIterable(self._X)

    def updateDataSet(self, X_new: pd.DataFrame, y_new: ndarray):
        self._X = X_new
        self._y = y_new

