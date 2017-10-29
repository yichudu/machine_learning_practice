from sklearn import multiclass
from sklearn import datasets,neighbors,linear_model,metrics
from sklearn.model_selection import train_test_split
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD


iris = datasets.load_iris()
print("The iris' target names: ", iris.target_names)
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)


y_train = keras.utils.to_categorical(y_train, num_classes=3)
y_test = keras.utils.to_categorical(y_test, num_classes=3)

model = Sequential()
model.add(Dense(10, activation='relu', input_dim=4))
model.add(Dropout(0.8))
model.add(Dense(5, activation='relu'))
model.add(Dense(3, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(X_train, y_train,
          epochs=50)
score = model.evaluate(X_test, y_test)
print('\n',model.metrics_names,'\n',score)
"""
 ['loss', 'acc'] 
 [0.88050086100896197, 0.75555555952919851]
"""