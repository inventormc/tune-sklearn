from sklearn.base import BaseEstimator
class TuneCV(BaseEstimator):
    def __init__(
            # see params in doc
            ):
        pass

    def _refit(self, X,  y=None, **fit_params):
        pass

    def fit(self, X, y=None, groups=None, **fit_params):
        pass

    def score(self, X, y=None):
        pass


class _Trainable(Trainable):
    def __init__(
            #see params in doc
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



