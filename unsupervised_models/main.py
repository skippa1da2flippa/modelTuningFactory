from classifier.gaussian_mixture import GaussianMixturesFactory
from utility.fetchedData import X_train, y_train
from utility.data_storer import DataHandler



gaussianFactory = GaussianMixturesFactory(X_train, y_train)

lst = gaussianFactory.modelsBuilder()

DataHandler.storeData(lst, r"experiment_result\data.pkl")


"""
res = DataHandler.getData(r"experiment_result\data.pkl")
maxim = 0
idx = -1

for x, dt in enumerate(res):
    if dt["rand-index"] > maxim:
        maxim = dt["rand-index"]
        idx = x

print()
"""









