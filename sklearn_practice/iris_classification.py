# use KNeighborsClassifier for multi-classification

from sklearn import multiclass
from sklearn import datasets,neighbors,linear_model,metrics
from sklearn.model_selection import train_test_split
import numpy as np


iris = datasets.load_iris()
print("The iris' target names: ", iris.target_names)
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)

clf=neighbors.KNeighborsClassifier()

clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
accuracy_score=metrics.accuracy_score(y_test,y_pred)
print("accuracy_score: {0}".format(accuracy_score))


"""
The iris' target names:  ['setosa' 'versicolor' 'virginica']
accuracy_score: 0.9111111111111111
"""



