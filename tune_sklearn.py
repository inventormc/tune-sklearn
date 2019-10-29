from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import cross_validate
from sklearn.model_selection import check_cv
from sklearn.base import is_classifier
from ray import tune
from ray.tune import Trainable
from ray.tune.schedulers import PopulationBasedTraining, MedianStoppingRule
from sklearn.base import clone

class TuneCV(BaseEstimator):
    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    @property
    def best_params_(self):
        check_is_fitted(self, "cv_results_")
        return self.cv_results_.best_params # Will need to modify this based off of `fit()`

    @property
    def best_score_(self):
        check_is_fitted(self, "cv_results_")
        return self.cv_results_.best_value

    @property
    def best_trial_(self):
        check_is_fitted(self, "cv_results_")
        return self.cv_results_.best_trial_

    @property
    def classes_(self):
        check_is_fitted(self, "cv_results_")
        return self.best_estimator_.classes_

    @property
    def n_trials_(self):
        return len(self.trials_)

    @property
    def trials_(self):
        check_is_fitted(self, "cv_results")
        return self.cv_results_.trials

    @property
    def decision_function(self):
        check_is_fitted(self, "cv_results_")
        return self.best_estimator_.decision_function

    @property
    def inverse_transform(self):
        check_is_fitted(self, "cv_results_")
        return self.best_estimator_.inverse_transform

    @property
    def predict(self):
        #check_is_fitted(self, "cv_results_")
        return self.best_estimator_.predict

    @property
    def predict_log_proba(self):
        check_is_fitted(self, "cv_results_")
        return self.best_estimator_.predict_log_proba

    @property
    def predict_proba(self):
        check_is_fitted(self, "cv_results_")
        return self.best_estimator_.predict_proba

    @property
    def transform(self):
        check_is_fitted(self, "cv_results_")
        return self.best_estimator_.transform

    def __init__(self, 
                 estimator, 
                 scheduler, 
                 param_grid, 
                 num_samples, 
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
        self.best_estimator_ = clone(self.estimator)

        self.best_estimator_.set_params(**fit_params)

        self.best_estimator_.fit(X, y, **fit_params)

    def fit(self, X, y=None, groups=None, **fit_params):
        scheduler = self.get_scheduler(self.scheduler)
        classifier = is_classifier(self.estimator)
        cv = check_cv(self.cv, y, classifier)
        config={}
        config['estimator'] = self.estimator
        config['scheduler'] = scheduler
        config['param_grid'] = self.param_grid
        config['X'] = X
        config['y'] = y
        config['groups'] = groups
        config['cv'] = cv
        config['fit_params'] = fit_params
        config['scoring'] = self.scoring
        analysis = tune.run(
                _Trainable,
                scheduler=MedianStoppingRule(),
                #scheduler=scheduler,
                reuse_actors=True,
                verbose=True,
                stop={"training_iteration":1},
                num_samples=self.num_samples,
                config=config
                )

        if self.refit:
            best_config = analysis.get_best_config(metric="test_accuracy")
            for key in ['estimator', 'scheduler', 'param_grid', 'X', 'y', 'groups', 'cv', 'fit_params', 'scoring']:
                best_config.pop(key)
            self._refit(X, y, **best_config)

        return self



    def score(self, X, y=None):
        pass

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
        self.param_grid = config.pop('param_grid')
        self.X = config.pop('X')
        self.y = config.pop('y')
        self.groups = config.pop('groups')
        self.cv = config.pop('cv')
        self.fit_params = config.pop('fit_params')
        self.scoring = config.pop('scoring')

        # print(config)

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

        return {
                "test_accuracy": sum(scores["test_score"])/len(scores["test_score"])
                }


    def _save(self, checkpoint_dir):
        return self.config

    def _restore(self, checkpoint):
        pass



