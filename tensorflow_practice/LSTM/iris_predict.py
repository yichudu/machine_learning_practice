import tensorflow as tf
import tensorflow.contrib.learn as learn
import tensorflow.contrib.layers as layers
import tensorflow.contrib.framework as framework

from sklearn import metrics
from sklearn import datasets
from sklearn import model_selection

import numpy as np
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat


def my_model(features, target):
    target=tf.one_hot(target,3,1,0)

    logits,loss=learn.models.logistic_regression(features, target)

    train_op = layers.optimize_loss(loss,
                                    framework.get_global_step(),
                                    optimizer='Adagrad',
                                    learning_rate=0.01
                                    )
    return tf.argmax(logits,1),loss,train_op

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = model_selection.train_test_split(iris.data, iris.target, test_size=0.2)

X_train, X_test = map(np.float32, [X_train, X_test])


classifier = SKCompat(learn.Estimator(model_fn=my_model, model_dir="target/Models/model_1"))
classifier.fit(X_train, y_train, steps=800)

y_predicted = [i for i in classifier.predict(X_test)]
score = metrics.accuracy_score(y_test, y_predicted)
print('Accuracy: {}%'.format(score * 100))
"""
Accuracy: 83.33%
"""