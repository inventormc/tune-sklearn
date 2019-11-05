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
    # TODO
    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    # TODO
    @property
    def best_params_(self):
        check_is_fitted(self, "cv_results_")
        return self.cv_results_.best_params # Will need to modify this based off of `fit()`

    # TODO
    @property
    def best_score_(self):
        check_is_fitted(self, "cv_results_")
        return self.cv_results_.best_value

    # TODO
    @property
    def best_trial_(self):
        check_is_fitted(self, "cv_results_")
        return self.cv_results_.best_trial_

    # TODO
    @property
    def classes_(self):
        check_is_fitted(self, "cv_results_")
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
        #return self.best_estimator_.predict

        return self.best_estimator.predict

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
    def __init__(self, estimator, scheduler, param_grid, num_samples, cv=3, refit=True, scoring=None
            ):
        self.estimator = estimator
        self.scheduler = scheduler
        self.num_samples = num_samples
        self.cv = cv
        self.scoring = scoring
        self.refit = refit
        self.param_grid = param_grid

    # TODO
    def _refit(self, X,  y=None, **fit_params):
        pass

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
        self.add_hyper_to_config(self.param_grid, config)
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
            best_config = analysis.get_best_config(metric="test accuracy")
            for key in ['estimator', 'scheduler', 'param_grid', 'X', 'y', 'groups', 'cv', 'fit_params', 'scoring']:
                best_config.pop(key)
            self.best_estimator = clone(self.estimator)

            self.best_estimator.set_params(**best_config)

            self.best_estimator.fit(X, y, **fit_params)


        return self



    # TODO
    def score(self, X, y=None):
        pass

    # TODO
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

    def add_hyper_to_config(self, param_grid, config):
        for name, distribution in self.param_grid.items():
            config[name] = tune.sample_from(lambda spec: distribution)


class _Trainable(Trainable):

    def _setup(self, config):
        self.estimator = clone(config['estimator'])
        self.scheduler = config['scheduler']
        self.param_grid = config['param_grid']
        self.X = config['X']
        self.y = config['y']
        self.groups = config['groups']
        self.cv = config['cv']
        self.fit_params = config['fit_params']
        self.scoring = config['scoring']

        for key in ['estimator', 'scheduler', 'param_grid', 'X', 'y', 'groups', 'cv', 'fit_params', 'scoring']:
            config.pop(key)

        print(config)

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
                "test accuracy": sum(scores["test_score"])/len(scores["test_score"])
                }


    def _save(self, checkpoint_dir):
        return self.config

    # TODO
    def _restore(self, checkpoint):
        pass



