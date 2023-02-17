from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
y = y.astype(int)
X = X / 255

# split the data between training and testing data (70-30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.99, random_state=109)

# split the data between training and testing data (70-30)
X_t2, X_test2, y_t2, y_test2 = train_test_split(X_test, y_test, test_size=0.99, random_state=109)

