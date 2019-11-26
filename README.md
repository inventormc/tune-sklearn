# tune-sklearn
Tune-sklearn is a package that integrates Ray Tune's hyperparameter tuning and scikit-learn's models, allowing users to optimize hyerparameter searching for sklearn using Tune's schedulers. Tune-sklearn follows the same API as scikit-learn's GridSearchCV, but allows for more flexibility in defining hyperparameter search regions, such as distributions to sample from.

## Quick Start
Use tune-sklearn TuneCV() to tune sklearn model

```python
iris = datasets.load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)

clf = SGDClassifier()
param_grid = {
    'n_estimators': randint(20, 80) #scipy stats randint
    #'n_estimators':[20, 30, 40, 50, 60, 70, 80] #you can also specify discrete values to gridsearch over as opposed to sampling
    'alpha': tune.sample_from(lambda spec: np.random.choice([1e-4, 1e-3, 1e-2, 1e-1]))
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
            n_jobs=5,
            refit=True,
            early_stopping=True,
            iters=10)
tune_search.fit(x_train, y_train)
```

## More information
[Ray Tune](https://ray.readthedocs.io/en/latest/tune.html)
