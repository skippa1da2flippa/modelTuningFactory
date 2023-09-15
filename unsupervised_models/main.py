from numpy import array
from classifier.gaussian_mixture import GaussianMixturesFactory
from unsupervised_models.classifier.normalized_cut import NormalizedCutFactory
from unsupervised_models.classifier.mean_shift import MeanShiftFactory
from unsupervised_models.utility.parallelization_handler import Parallelize
from utility.fetchedData import X_train, y_train
from utility.data_storer import DataHandler

"""
gaussianFactory = GaussianMixturesFactory(X_train, y_train)
lst = gaussianFactory.modelsBuilder()
DataHandler.storeData(lst, r"experiment_result\gaussian_data.pkl")
"""

"""
    res = DataHandler.getData(r"experiment_result\gaussian_data.pkl")
maxim = 0
idx = -1

for x, dt in enumerate(res):
    if dt["rand-index"] > maxim:
        maxim = dt["rand-index"]
        idx = x

print()
"""


def gaussianBuilder():
    gaussianFactory = GaussianMixturesFactory(X_train, y_train)
    lst = gaussianFactory.modelsBuilder()
    DataHandler.storeData(lst, r"experiment_result\gaussian_data.pkl")
    print(f"\033[Just completed gaussian mixtures\033[0m")


def nCutBuilder():
    norm = NormalizedCutFactory(X_train, y_train)
    lst = norm.modelsBuilder()
    DataHandler.storeData(lst, r"experiment_result\nCut_data.pkl")
    print(f"\033[Just completed ncut\033[0m")


def meanBuilder():
    mean = MeanShiftFactory(X_train, y_train)
    lst = mean.modelsBuilder()
    DataHandler.storeData(lst, r"experiment_result\mean_data.pkl")
    print(f"\033[Just completed mean shift\033[0m")


if __name__ == '__main__':
    gaussian = Parallelize.parallelizeOneWithOne(gaussianBuilder)
    nCut = Parallelize.parallelizeOneWithOne(nCutBuilder)
    meanShift = Parallelize.parallelizeOneWithOne(meanBuilder)

    Parallelize.waitProcesses(array([gaussian, nCut, meanShift]))
