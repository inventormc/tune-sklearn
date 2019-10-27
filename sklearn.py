from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from ray.tune import Trainable

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
        check_is_fitted(self, "cv_results_")
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
    
    def __init__(self, estimator, param_grid
            # and other optional params in doc
            ):
        pass

    def _refit(self, X,  y=None, **fit_params):
        pass

    def fit(self, X, y=None, groups=None, **fit_params):
        pass

    def score(self, X, y=None):
        pass


class _Trainable(Trainable):
    def __init__(self, estimator, param_grid, X, y
            # and other optional params in doc
            ):
        pass

    def _setup(self, config):
        pass

    def _train(self):
        pass

    def _save(self, checkpoint_dir):
        pass

    def _restore(self, checkpoint):
        pass



