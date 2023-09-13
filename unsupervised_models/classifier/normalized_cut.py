import pandas as pd
from numpy import ndarray
from sklearn.cluster import SpectralClustering
from unsupervised_models.base_classes.base_class import BaseModelFactory

"""
    Factory implementation
"""


class NormalizedCutFactory(BaseModelFactory):

    def __init__(self, X: pd.DataFrame, y: ndarray):
        # save the data set
        self._X: pd.DataFrame = X
        self._y: ndarray = y

        # initialize the possible candidates (number of clusters)
        self.nClusters: list[int] = [x for x in range(5, 16)]

        # generate k normalized cut models each of which with a different number of clusters
        self.meanShiftModel: list[SpectralClustering] = [
            SpectralClustering(
                n_clusters=nCluster
            ) for nCluster in self.nClusters
        ]

        # define what sequence of actions the extended class should do
        def _fitNdTesting(normalizedCutModel: SpectralClustering, X_t: pd.DataFrame):
            # train the model and return how the model doing with the training data
            return normalizedCutModel.fit_predict(X_t)

        # set up the base class
        super().__init__(
            self.meanShiftModel, self.nClusters, _fitNdTesting, self._X, self._y
        )

