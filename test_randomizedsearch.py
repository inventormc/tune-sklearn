from tune_sklearn import TuneRandomizedSearchCV, TuneGridSearchCV
from scipy.stats import randint, uniform
from sklearn.model_selection import GridSearchCV
from ray import tune
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from ray.tune.schedulers import PopulationBasedTraining, MedianStoppingRule
import random
import unittest

# TODO: Either convert to individual examples or to python unittests

class RandomizedSearchTest(unittest.TestCase):
    def test_random_forest(self):
        iris = datasets.load_iris()
        x = iris.data
        y = iris.target
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)

        clf = RandomForestClassifier()
        param_grid = {
            'n_estimators': randint(20,80)
        }


        tune_search = TuneRandomizedSearchCV(clf, param_grid, scheduler=MedianStoppingRule(), iters=20)
        tune_search.fit(x_train, y_train)

        pred = tune_search.predict(x_test)
        print(pred)
        accuracy = np.count_nonzero(np.array(pred) == np.array(y_test))/len(pred)
        print(accuracy)

    def test_pbt(self):
        iris = datasets.load_iris()
        x = iris.data
        y = iris.target
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)

        clf = SGDClassifier()
        param_distributions = {
            #'alpha': tune.sample_from(uniform(1e-4, 1e-1))
            'alpha': np.random.uniform(1e-4, 1e-1)
        }
        # param_grid = {
        #     'alpha': [1e-4, 1e-3, 1e-2, 1e-1]
        # }

        scheduler = PopulationBasedTraining(
                    time_attr="training_iteration",
                    metric="average_test_score",
                    mode="max",
                    perturbation_interval=5,
                    resample_probability=1.0,
                    hyperparam_mutations = {
                        "alpha" : lambda: np.random.uniform()
                    })

        tune_search = TuneRandomizedSearchCV(clf,
                    param_distributions,
                    scheduler=scheduler,
                    early_stopping=True,
                    iters=10,
                    verbose=1,
                    num_samples=3,
                    )
        # tune_search = TuneGridSearchCV(clf,
        #             param_grid,
        #             refit=True,
        #             early_stopping=True,
        #             iters=10,
        #             verbose=1,
        #             )
        tune_search.fit(x_train, y_train)

        pred = tune_search.predict(x_test)
        print(pred)
        accuracy = np.count_nonzero(np.array(pred) == np.array(y_test))/len(pred)
        print(accuracy)
        print(tune_search.best_params_)

    def test_linear_iris(self):
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target

        logistic = linear_model.LogisticRegression()

        # Create regularization penalty space
        penalty = ['l1', 'l2']

        # Create regularization hyperparameter space
        C = np.logspace(0, 4, 5)

        # Create hyperparameter options
        hyperparameters = dict(C=C, penalty=penalty)

        clf = TuneRandomizedSearchCV(logistic, hyperparameters, scheduler=MedianStoppingRule())
        clf.fit(X,y)

        pred = clf.predict(X)
        print(pred)
        accuracy = np.count_nonzero(np.array(pred) == np.array(y))/len(pred)
        print(accuracy)


if __name__ == '__main__':
    unittest.main()

# TODO: Edit these tests from sklearn once we have properties/all signatures finished in tune_sklearn

'''
def test_grid_search():
    pass

def test_grid_search_with_fit_params():
    pass

def test_random_search_with_fit_params():
    pass

def test_grid_search_no_score():
    pass

def test_grid_search_score_method():
    pass

def test_grid_search_groups():
    pass

def test_no_refit():
    pass

def test_grid_search_error():
    pass

def test_grid_search_one_grid_point():
    pass
'''

