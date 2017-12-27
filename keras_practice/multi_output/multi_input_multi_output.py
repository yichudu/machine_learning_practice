"""
与 iris_classification.py 对比, 是两种不同的模型结构.
前者是序列模型, 本源文件是通用模型. 在处理像TextCNN这种先分支再融合的网络, 通用模型更灵活.
"""
from sklearn import multiclass
from sklearn import datasets,neighbors,linear_model,metrics
from sklearn.model_selection import train_test_split
from keras.layers.advanced_activations import PReLU
import numpy as np
from keras.utils import plot_model
from keras.layers import BatchNormalization

import keras
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation,Input,concatenate
from keras.optimizers import SGD
from keras import backend
from keras.utils.generic_utils import get_custom_objects

FEATURE_DIM=4
OUTPUT_NUM=3
INPUT_NUM=2
iris = datasets.load_iris()


def get_multi_input_tensor():
    input_tensor_arr=[]
    for i in range(INPUT_NUM):
        input_tensor = Input(shape=(FEATURE_DIM,), dtype='float32', name='random_input_{}'.format(i))
        input_tensor_arr.append(input_tensor)
    return input_tensor_arr


def get_multi_output_tensor(shared_layer):
    output_arr=[]
    for i in range(OUTPUT_NUM):
        inputs=shared_layer
        dense1=Dense(10, activation='relu', input_dim=4)(inputs)
        batch_norm=BatchNormalization()(dense1)
        dense2=Dense(1, input_dim=4,name='y_{}'.format(i))(batch_norm)
        output_arr.append(dense2)
    return output_arr


def get_model():
    input_tensor_arr=get_multi_input_tensor()
    merge_input_tensor_arr=[]
    for input_tensor in input_tensor_arr:
        dense1 = Dense(10, activation='relu', input_dim=4)(input_tensor)
        dense2 = Dense(10)(dense1)
        merge_input_tensor_arr.append(dense2)
    concatenate_layer=concatenate(merge_input_tensor_arr)
    model = Model(inputs=input_tensor_arr, outputs=get_multi_output_tensor(concatenate_layer))
    return model

model=get_model() # type:keras.models.Model
plot_model(model, to_file='model.png',show_shapes='True')

def get_data():
    input_data_0 = np.random.randint(low=1, high=10, size=[1000, FEATURE_DIM])
    input_data_1 = np.random.randint(low=5, high=8, size=[1000, FEATURE_DIM])
    output_data_arr=[]
    for i in range(OUTPUT_NUM):
        output_data_arr.append(np.random.randint(low=1,high=10,size=[1000,1]))

    X0_train, X0_test, X1_train, X1_test, y0_train, y0_test, y1_train, y1_test, y2_train, y2_test = train_test_split(input_data_0,input_data_1,output_data_arr[0],output_data_arr[1],output_data_arr[2], test_size=0.3)
    return X0_train, X0_test, X1_train,X1_test,  y0_train, y0_test, y1_train, y1_test, y2_train, y2_test

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss=['mean_squared_error']*3,
              optimizer=sgd,
              metrics=['mean_absolute_error'])
X0_train, X0_test, X1_train,X1_test,  y0_train, y0_test, y1_train, y1_test, y2_train, y2_test=get_data()
model.fit( x=[X0_train,X1_train], y=[y0_train,y1_train,y2_train], validation_data=([X0_test,X1_test],[y0_test,y1_test,y2_test]),
          epochs=10,verbose=2)
"""
Epoch 3/10
0s - loss: 20.8063 - y_0_loss: 6.9352 - y_1_loss: 6.7669 - y_2_loss: 7.1042 - y_0_mean_absolute_error: 2.2661 - y_1_mean_absolute_error: 2.2515 - y_2_mean_absolute_error: 2.3238 - 
val_loss: 25.1368 - val_y_0_loss: 6.2392 - val_y_1_loss: 8.9462 - val_y_2_loss: 9.9514 - val_y_0_mean_absolute_error: 2.1454 - val_y_1_mean_absolute_error: 2.5280 - val_y_2_mean_absolute_error: 2.5901
"""