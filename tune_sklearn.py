from scipy.stats import _distn_infrastructure, rankdata
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import cross_validate, ParameterGrid
from sklearn.model_selection import check_cv
from sklearn.metrics.scorer import check_scoring
from sklearn.base import is_classifier
from sklearn.utils.metaestimators import _safe_split
from sklearn.base import clone
from ray import tune
from ray.tune import Trainable
from ray.tune.schedulers import PopulationBasedTraining, MedianStoppingRule
import numpy as np
import os
import pickle

class TuneCV(BaseEstimator):
    # TODO
    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    # TODO
    @property
    def best_params_(self):
        return self.best_estimator_.best_params

    # TODO
    @property
    def best_score_(self):
        return self.best_estimator_.best_value

    # TODO
    @property
    def best_trial_(self):
        check_is_fitted(self, "cv_results_")
        return self.cv_results_.best_trial_

    # TODO
    @property
    def classes_(self):
        return self.best_estimator_.classes_

    # TODO
    @property
    def n_trials_(self):
        return len(self.trials_)

    # TODO
    @property
    def trials_(self):
        check_is_fitted(self, "cv_results")
        return self.cv_results_.trials

    # TODO
    @property
    def decision_function(self):
        check_is_fitted(self, "cv_results_")
        return self.best_estimator_.decision_function

    # TODO
    @property
    def inverse_transform(self):
        check_is_fitted(self, "cv_results_")
        return self.best_estimator_.inverse_transform

    # TODO
    @property
    def predict(self):
        #check_is_fitted(self, "cv_results_")
        return self.best_estimator_.predict

    # TODO
    @property
    def predict_log_proba(self):
        check_is_fitted(self, "cv_results_")
        return self.best_estimator_.predict_log_proba

    # TODO
    @property
    def predict_proba(self):
        check_is_fitted(self, "cv_results_")
        return self.best_estimator_.predict_proba

    # TODO
    @property
    def transform(self):
        check_is_fitted(self, "cv_results_")
        return self.best_estimator_.transform
    
    def _check_params(self):
        if not hasattr(self.estimator, 'fit'):
            raise ValueError('estimator must be a scikit-learn estimator.')

        if type(self.param_grid) is not dict:
            raise ValueError('param_distributions must be a dictionary.')

        if self.early_stopping and not hasattr(self.estimator, 'partial_fit'):
            raise ValueError('estimator must support partial_fit.')

    # TODO: Add all arguments found in optuna to constructor
    def __init__(self,
                 estimator,
                 param_grid,
                 scheduler=None,
                 scoring=None,
                 n_jobs=None,
                 cv=3,
                 refit=True,
                 verbose=0,
                 error_score='raise',
                 return_train_score=False,
                 early_stopping=False,
                 iters=None,
    ):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scheduler = scheduler
        self.num_samples = n_jobs
        self.cv = cv
        self.scoring = check_scoring(estimator, scoring)
        self.refit = refit
        self.verbose = verbose
        self.error_score = error_score
        self.return_train_score = return_train_score
        self.early_stopping = early_stopping
        self.iters = iters

    def _refit(self, X, y=None, **fit_params):
        self.best_estimator_ = clone(self.estimator)
        self.best_estimator_.set_params(**self.best_params)
        self.best_estimator_.fit(X, y, **fit_params)

    def fit(self, X, y=None, groups=None, **fit_params):
        self._check_params()
        classifier = is_classifier(self.estimator)
        cv = check_cv(self.cv, y, classifier)
        n_splits = cv.get_n_splits(X, y, groups)
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)
        config={}
        hyperparams = self.param_grid
        for key, distribution in hyperparams.items():
            if isinstance(distribution, (list, np.ndarray)):
                config[key] = tune.grid_search(list(distribution))
            elif isinstance(distribution, (_distn_infrastructure.rv_frozen)):
                config[key] = tune.sample_from(lambda spec: distribution.rvs(1)[0])
            else:
                config[key] = tune.sample_from(lambda spec: distribution)

        config['scheduler'] = self.scheduler
        config['X'] = X
        config['y'] = y
        config['groups'] = groups
        config['cv'] = cv
        config['fit_params'] = fit_params
        config['scoring'] = self.scoring
        config['early_stopping'] = self.early_stopping
        if self.early_stopping:
            config['estimator'] = [clone(self.estimator) for _ in range(cv.get_n_splits(X, y))]
            config['iters'] = self.iters
            analysis = tune.run(
                    _Trainable,
                    scheduler=self.scheduler,
                    reuse_actors=True,
                    verbose=self.verbose,
                    stop={"training_iteration":self.iters},
                    num_samples=self.num_samples,
                    config=config,
                    checkpoint_at_end=True
                    )
        else:
            config['estimator'] = self.estimator
            analysis = tune.run(
                    _Trainable,
                    scheduler=self.scheduler,
                    reuse_actors=True,
                    verbose=self.verbose,
                    stop={"training_iteration":1},
                    num_samples=self.num_samples,
                    config=config,
                    checkpoint_at_end=True
                    )
        
        candidate_params = list(ParameterGrid(self.param_grid))
        self.cv_results_ = self._format_results(candidate_params, self.scorer_, n_splits, analysis)

        best_config = analysis.get_best_config(metric="average_test_score", mode="max")
        for key in ['estimator', 'scheduler', 'X', 'y', 'groups', 'cv', 'fit_params', 'scoring', 'early_stopping', 'iters']:
            best_config.pop(key)
        self.best_params = best_config
        if self.refit:
            self._refit(X, y, **fit_params)

        return self

    def score(self, X, y=None):
        return self.scorer_(self.best_estimator_, X, y)
    
    def _format_results(self, candidate_params, scorers, n_splits, out):
        # TODO: Extract relevant parts out of `analysis` object from Tune
        if self.return_train_score:
            fit_times, test_scores, score_times, train_scores = zip(*out)
        else:
            fit_times, test_scores, score_times = zip(*out)
        
        results = {"params": candidate_params}
        n_candidates = len(candidate_params)

        def _store(
            results,
            key_name,
            array,
            n_splits,
            n_candidates,
            weights=None,
            splits=False,
            rank=False,
        ):
            """A small helper to store the scores/times to the cv_results_"""
            # When iterated first by n_splits and then by parameters
            array = np.array(array, dtype=np.float64).reshape(n_splits, n_candidates)
            if splits:
                for split_i in range(n_splits):
                    results["split%d_%s" % (split_i, key_name)] = array[:, split_i]

            array_means = np.average(array, axis=1, weights=weights)
            results["mean_%s" % key_name] = array_means
            # Weighted std is not directly available in numpy
            array_stds = np.sqrt(
                np.average((array - array_means[:, np.newaxis]) ** 2, axis=1, weights=weights)
            )
            results["std_%s" % key_name] = array_stds

            if rank:
                results["rank_%s" % key_name] = np.asarray(
                    rankdata(-array_means, method="min"), dtype=np.int32
                )
        
        _store(results, 'fit_time', fit_times, n_splits, n_candidates)
        _store(results, 'score_time', score_times, n_splits, n_candidates)
        _store(results, "test_score", test_scores, n_splits, n_candidates, splits=True, rank=True)
        if self.return_train_score:
            _store(results, "train_score", train_scores, n_splits, n_candidates, splits=True)

        return results

    # We may not need this function if the user passes in the actual scheduler
    # but then they need to follow this syntax for the hyperparam_mutations
    def get_scheduler(self, scheduler):
        if scheduler == "pbt":
            return PopulationBasedTraining(
                time_attr="training_iteration",
                metric="mean_accuracy",
                mode="max",
                perturbation_interval=20,
                hyperparam_mutations = {
                    name: lambda: distribution for name, distribution in self.param_grid.items()
                }
            )


class _Trainable(Trainable):

    def _setup(self, config):

        self.estimator = clone(config.pop('estimator'))
        self.scheduler = config.pop('scheduler')
        self.X = config.pop('X')
        self.y = config.pop('y')
        self.groups = config.pop('groups')
        self.fit_params = config.pop('fit_params')
        self.scoring = config.pop('scoring')
        self.early_stopping = config.pop('early_stopping')
        self.iters = config.pop('iters')
        self.cv = config.pop('cv')

        self.estimator_config = config

        if self.early_stopping:
            n_splits = self.cv.get_n_splits(self.X, self.y)
            self.fold_scores = np.zeros(n_splits)
            for i in range(n_splits):
                self.estimator[i].set_params(**self.estimator_config)
        else:
            self.estimator.set_params(**self.estimator_config)

    def _train(self):
        if self.early_stopping:
            for i, (train, test) in enumerate(self.cv.split(self.X, self.y)):
                X_train, y_train = _safe_split(self.estimator, self.X, self.y, train)
                X_test, y_test = _safe_split(
                    self.estimator,
                    self.X,
                    self.y,
                    test,
                    train_indices=train
                )
            self.estimator[i].partial_fit(X_train, y_train, np.unique(self.y))
            self.fold_scores[i] = self.scoring(self.estimator[i], X_test, y_test)

            self.mean_scores = sum(self.fold_scores)/len(self.fold_scores)
            return {"average_test_score": self.mean_scores}
        else:
            scores = cross_validate(
                self.estimator,
                self.X,
                self.y,
                cv=self.cv,
                fit_params=self.fit_params,
                groups=self.groups,
                scoring=self.scoring
            )

            self.test_accuracy = sum(scores["test_score"])/len(scores["test_score"])
            return {"average_test_score": self.test_accuracy}

    def _save(self, checkpoint_dir):
        path = os.path.join(checkpoint_dir, "checkpoint")
        with open(path, 'wb') as f:
            pickle.dump(self.estimator, f)
        return path

    def _restore(self, checkpoint):
        with open(checkpoint, 'rb') as f:
            self.estimator = pickle.load(f)

    def reset_config(self, new_config):
        return True



