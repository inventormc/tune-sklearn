from tune_sklearn import TuneRandomizedSearchCV, TuneGridSearchCV
from scipy.stats import randint, uniform
from sklearn.model_selection import GridSearchCV
from ray import tune
import numpy as np
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from ray.tune.schedulers import MedianStoppingRule
import unittest

class GridSearchTest(unittest.TestCase):
    def test_digits(self):
        # Loading the Digits dataset
        digits = datasets.load_digits()

        # To apply an classifier on this data, we need to flatten the image, to
        # turn the data in a (samples, feature) matrix:
        n_samples = len(digits.images)
        X = digits.images.reshape((n_samples, -1))
        y = digits.target

        # Split the dataset in two equal parts
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=0)

        # Set the parameters by cross-validation
        tuned_parameters = {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                             'C': [1, 10, 100, 1000]}

        tune_search = TuneGridSearchCV(SVC(),
                               tuned_parameters,
                               scheduler=MedianStoppingRule(),
                               iters=20)
        tune_search.fit(X_train, y_train)

        pred = tune_search.predict(X_test)
        print(pred)
        accuracy = np.count_nonzero(np.array(pred) == np.array(y_test))/len(pred)
        print(accuracy)

    def test_diabetes(self):
        # load the diabetes datasets
        dataset = datasets.load_diabetes()
        X = dataset.data
        y = dataset.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=0)
        # prepare a range of alpha values to test
        alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
        param_grid = dict(alpha=alphas)
        # create and fit a ridge regression model, testing each alpha
        model = linear_model.Ridge()

        tune_search = TuneGridSearchCV(model, param_grid, MedianStoppingRule())
        tune_search.fit(X_train, y_train)

        pred = tune_search.predict(X_test)
        print(pred)
        error = sum(np.array(pred) - np.array(y_test))/len(pred)
        print(error)


if __name__ == '__main__':
    unittest.main()
