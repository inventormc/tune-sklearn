from scipy.stats import _distn_infrastructure
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import cross_validate
from sklearn.model_selection import check_cv
from sklearn.base import is_classifier
from ray import tune
from ray.tune import Trainable
from ray.tune.schedulers import PopulationBasedTraining, MedianStoppingRule
from sklearn.base import clone
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

    # TODO: Add all arguments found in optuna to constructor
    def __init__(self,
                 estimator,
                 scheduler,
                 param_grid=None,
                 num_samples=3,
                 cv=3,
                 refit=True,
                 scoring=None,
    ):
        self.estimator = estimator
        self.scheduler = scheduler
        self.num_samples = num_samples
        self.cv = cv
        self.scoring = scoring
        self.refit = refit
        self.param_grid = param_grid

    def _refit(self, X, y=None, **fit_params):
        self.best_estimator_.fit(X, y, **fit_params)

    def _partial_fit_and_early_stop(self,):

    def fit(self, X, y=None, groups=None, **fit_params):
        classifier = is_classifier(self.estimator)
        cv = check_cv(self.cv, y, classifier)
        config={}
        hyperparams = self.param_grid
        for key, distribution in hyperparams.items():
            if isinstance(distribution, (list, np.ndarray)):
                config[key] = tune.grid_search(list(distribution))
            elif isinstance(distribution, (_distn_infrastructure.rv_frozen)):
                config[key] = tune.sample_from(lambda spec: distribution.rvs(1)[0])
            else:
                config[key] = tune.sample_from(lambda spec: distribution)

        config['estimator'] = self.estimator
        config['scheduler'] = self.scheduler
        config['X'] = X
        config['y'] = y
        config['groups'] = groups
        config['cv'] = cv
        config['fit_params'] = fit_params
        config['scoring'] = self.scoring
        analysis = tune.run(
                _Trainable,
                scheduler=self.scheduler,
                reuse_actors=True,
                verbose=True, # silence in the future
                stop={"training_iteration":1},
                num_samples=self.num_samples,
                config=config
                )

        best_config = analysis.get_best_config(metric="test_accuracy")
        for key in ['estimator', 'scheduler', 'X', 'y', 'groups', 'cv', 'fit_params', 'scoring']:
            best_config.pop(key)
        self.best_params = best_config
        self.best_estimator_ = clone(self.estimator)
        self.best_estimator_.set_params(**self.best_params)
        if self.refit:
            self._refit(X, y, **fit_params)
        else:
            logdir = analysis.get_best_logdir(metric="test_accuracy", mode="max")
            with open(os.path.join(logdir, "checkpoint"), 'rb') as f:
                self.best_estimator  = pickle.load(f)

        return self



    # TODO
    def score(self, X, y=None):
        pass

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
        self.cv = config.pop('cv')
        self.fit_params = config.pop('fit_params')
        self.scoring = config.pop('scoring')

        self.estimator_config = config




    def _train(self):
        self.estimator.set_params(**self.estimator_config)
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

        return {
                "test_accuracy": self.test_accuracy
                }


    def _save(self, checkpoint_dir):
        with open(os.path.join(checkpoint_dir, "checkpoint"), 'wb') as f:
            pickle.dump(self.estimator, f)
        return checkpoint_dir

    def _restore(self, checkpoint):
        with open(os.path.join(checkpoint, "checkpoint"), 'rb') as f:
            self.estimator = pickle.load(f)



