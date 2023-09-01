import pandas as pd
from numpy import ndarray
from sklearn.cluster import MeanShift
from unsupervised_models.base_classes.base_class import BaseModelFactory

"""
    Factory implementation
"""


class MeanShiftFactory(BaseModelFactory):

    def __init__(self, X: pd.DataFrame, y: ndarray):
        # save the data set
        self._X: pd.DataFrame = X
        self._y: ndarray = y

        # initialize the possible candidates (kernel width)
        self.kernelWidths: list[float] = [1.25, 9.5, 30.75, 50.25, 70.5, 100.75, 200.25, 350.5]

        # generate k mean shift models each of which with a different kernel size
        self.meanShiftModel: list[MeanShift] = [
            MeanShift(
                bandwidth=kernelWidth
            ) for kernelWidth in self.kernelWidths
        ]

        # define what sequence of actions the extended class should do
        def _fitNdTesting(meanShiftModel: MeanShift, X_t: pd.DataFrame):
            # train the model
            meanShiftModel.fit(X_t)

            # return how the model doing with the training data
            return meanShiftModel.predict(X_t)

        # set up the base class
        super().__init__(
            self.meanShiftModel, self.kernelWidths, _fitNdTesting, self._X, self._y
        )
