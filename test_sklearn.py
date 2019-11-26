from tune_sklearn import TuneCV
from scipy.stats import randint
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

# TODO: Either convert to individual examples or to python unittests

class MockClassifier:
    """Dummy classifier to test the parameter search algorithms"""
    def __init__(self, foo_param=0):
        self.foo_param = foo_param

    def fit(self, X, Y):
        assert len(X) == len(Y)
        self.classes_ = np.unique(Y)
        return self

    def predict(self, T):
        return T.shape[0]

    def transform(self, X):
        return X + self.foo_param

    def inverse_transform(self, X):
        return X - self.foo_param

    predict_proba = predict
    predict_log_proba = predict
    decision_function = predict

    def score(self, X=None, Y=None):
        if self.foo_param > 1:
            score = 1.
        else:
            score = 0.
        return score

    def get_params(self, deep=False):
        return {'foo_param': self.foo_param}

    def set_params(self, **params):
        self.foo_param = params['foo_param']
        return self


class LinearSVCNoScore(LinearSVC):
    """An LinearSVC classifier that has no score method."""
    @property
    def score(self):
        raise AttributeError

def random_forest():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)

    clf = RandomForestClassifier()
    param_grid = {
        'n_estimators': tune.sample_from(lambda spec: random.randint(20,80))
    }


    tune_search = TuneCV(clf, MedianStoppingRule(), param_grid, 20)
    tune_search.fit(x_train, y_train)

    pred = tune_search.predict(x_test)
    print(pred)
    accuracy = np.count_nonzero(np.array(pred) == np.array(y_test))/len(pred)
    print(accuracy)

def pbt():
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)

    clf = SGDClassifier()
    param_grid = {
        #'n_estimators': randint(20, 80)
        'alpha': [1e-4, 1e-3, 1e-2, 1e-1]
    }

    scheduler = PopulationBasedTraining(
                time_attr="training_iteration",
                metric="average_test_score",
                mode="max",
                perturbation_interval=5,
                resample_probability=1.0,
                hyperparam_mutations = {
                    "alpha" : lambda: np.random.choice([1e-4, 1e-3, 1e-2, 1e-1])
                })

    tune_search = TuneCV(clf, 
                param_grid=param_grid,
                refit=True,
                early_stopping=True,
                iters=10,
                verbose=0,
                )
    tune_search.fit(x_train, y_train)

    pred = tune_search.predict(x_test)
    print(pred)
    accuracy = np.count_nonzero(np.array(pred) == np.array(y_test))/len(pred)
    print(accuracy)
    print(tune_search.best_params_)

def linear_iris():
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

    clf = TuneCV(logistic, MedianStoppingRule(), hyperparameters)
    clf.fit(X,y)

    pred = clf.predict(X)
    print(pred)
    accuracy = np.count_nonzero(np.array(pred) == np.array(y))/len(pred)
    print(accuracy)

def digits():
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

    tune_search = TuneCV(SVC(), MedianStoppingRule(), tuned_parameters, 20)
    tune_search.fit(X_train, y_train)

    pred = tune_search.predict(X_test)
    print(pred)
    accuracy = np.count_nonzero(np.array(pred) == np.array(y_test))/len(pred)
    print(accuracy)

def diabetes():
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

    tune_search = TuneCV(model, MedianStoppingRule(), param_grid)
    tune_search.fit(X_train, y_train)

    pred = tune_search.predict(X_test)
    print(pred)
    error = sum(np.array(pred) - np.array(y_test))/len(pred)
    print(error)


# TODO: Edit these tests from sklearn once we have properties/all signatures finished in tune_sklearn

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



#random_forest()
#linear_iris()
pbt()
#digits()
#diabetes()
