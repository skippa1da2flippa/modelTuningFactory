import pandas as pd
from numpy import ndarray
from sklearn.mixture import GaussianMixture
from sklearn.mixture._base import BaseMixture
from unsupervised_models.base_classes.base_class import BaseModelFactory

"""
    Factory implementation
"""


class GaussianMixturesFactory(BaseModelFactory):

    def __init__(self, X: pd.DataFrame, y: ndarray):
        # save the data set
        self._X: pd.DataFrame = X
        self._y: ndarray = y

        # initialize the possible candidates (number of clusters)
        self.nClusters: list[int] = [x for x in range(1, 16)]

        # generate k gaussian mixtures each of which with a different number of clusters
        self.gaussianMixtures: list[BaseMixture] = [
            GaussianMixture(
                n_components=n_classes, covariance_type="diag", random_state=1629
            ) for n_classes in self.nClusters
        ]

        # define what sequence of actions the extended class should do
        def _fitNdTesting(gaussianMixture: BaseMixture, X_t: pd.DataFrame):
            # train the model
            gaussianMixture.fit(X_t)

            # return how the model doing with the training data
            return gaussianMixture.predict(X_t)

        # set up the base class
        super().__init__(
            self.gaussianMixtures, self.nClusters, _fitNdTesting, self._X, self._y
        )

    def tuningProcess(self, dataset: pd.DataFrame, returnBestModel: bool = False) -> tuple:
        return super().tuningProcess(dataset, returnBestModel)

    def updateDataSet(self, X_new: pd.DataFrame, y_new: ndarray):
        self._X = X_new
        self._y = y_new
        super().updateDataSet(X_new, y_new)
