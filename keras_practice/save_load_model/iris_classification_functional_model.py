# coding:utf-8

"""
与 iris_classification.py 对比, 是两种不同的模型结构.
前者是序列模型, 本源文件是通用模型. 在处理像TextCNN这种先分支再融合的网络, 通用模型更灵活.
"""
import os
from sklearn import multiclass
import tensorflow as tf
from sklearn import datasets,neighbors,linear_model,metrics
from sklearn.model_selection import train_test_split
from keras.layers.advanced_activations import PReLU
import numpy as np
#from keras.utils import plot_model


import keras
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation,Input
from keras.optimizers import SGD
from keras.models import load_model

FEATURE_DIM=4
iris = datasets.load_iris()
print("The iris' target names: ", iris.target_names)
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)


y_train = keras.utils.to_categorical(y_train, num_classes=3)
y_test = keras.utils.to_categorical(y_test, num_classes=3)

inputs=Input(shape=(FEATURE_DIM,),dtype='float32',name='iris_data_input')
dense1=Dense(10, activation='relu', input_dim=4)(inputs)
dense2=Dense(5,activation='relu')(dense1)
outpus=Dense(3, activation='softmax')(dense2)

model=Model(inputs=inputs,outputs=outpus)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
#plot_model(model, to_file='model.png',show_shapes='True')
early_stop_callback=keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
model.fit(X_train, y_train,validation_data=(X_test,y_test),
          epochs=50,callbacks=[early_stop_callback],verbose=2)
evaluate_score = model.evaluate(X_test, y_test)
print('\n', model.metrics_names,'\n', evaluate_score)

FLAGS = tf.app.flags.FLAGS
ckp_path = os.path.join(FLAGS.buckets, "yichu_20180224_my_model.h5")

model.save(ckp_path)  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model
print('\n\n save_load model \n\n')
# returns a compiled model
# identical to the previous one
model = load_model('my_model.h5')
evaluate_score = model.evaluate(X_test, y_test)
print('\n', model.metrics_names,'\n', evaluate_score)

"""
32/45 [====================>.........] - ETA: 0s
 ['loss', 'acc'] 
 [0.39104276763068307, 0.8888889087571038]


 save_load model 


32/45 [====================>.........] - ETA: 0s
 ['loss', 'acc'] 
 [0.39104276763068307, 0.8888889087571038]
"""