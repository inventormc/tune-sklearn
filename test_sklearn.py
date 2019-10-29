from tune_sklearn import TuneCV
from ray import tune
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import random

iris = load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)

clf = RandomForestClassifier()
param_grid = {
    'n_estimators': tune.sample_from(lambda spec: random.randint(20,80))
}


tune_search = TuneCV(clf, "pbt", param_grid, 5)
tune_search.fit(x_train, y_train)

print(tune_search.predict(x_test))
print(y_test)
