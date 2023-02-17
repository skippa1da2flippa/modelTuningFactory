from classifier.gaussian_mixture import GaussianMixturesFactory
from utility.fetchedData import X_train, y_train

gaussianFactory = GaussianMixturesFactory(X_train, y_train)

lst = gaussianFactory.modelsBuilder()

for dicty in lst:
    print(dicty)







