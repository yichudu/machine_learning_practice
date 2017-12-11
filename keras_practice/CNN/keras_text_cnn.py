"""
https://github.com/bhaveshoswal/CNN-text-classification-keras/blob/master/model.py
"""
import pandas as pd
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from sklearn.model_selection import train_test_split
import keras
from keras.layers.embeddings import Embedding
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Reshape
from  keras.layers.convolutional import Conv2D, MaxPooling2D, Convolution2D
from keras.layers.recurrent import Recurrent
from keras.optimizers import SGD,Adam
import numpy as np
from keras.layers import merge, concatenate
from keras.utils import plot_model

import keras_practice.NLP.keras_nlp as keras_nlp
import pydot_ng


class Mydata:
    def __init__(self):
        self.VOCAB_SIZE = len(keras_nlp.tk.word_index)
        self.WORD_VEC_DIM = 5
        # 单个文档的最大截断长度
        self.MAX_TEXT_LEN = keras_nlp.MAX_LEN


mydata = Mydata()
tk = keras_nlp.tk


def get_embedding_matrix():
    embedding_matrix = np.zeros(shape=[mydata.VOCAB_SIZE + 1, mydata.WORD_VEC_DIM])  # 把index=0的空出来
    for word, index in tk.word_index.items():
        vec = np.random.standard_normal(size=(mydata.WORD_VEC_DIM,))
        embedding_matrix[index] = vec
    return embedding_matrix


X_train = keras_nlp.text_pad_sequences
y_train = np.random.uniform(1, 3, size=len(X_train))
y_train = np.random.choice(range(10),size=len(X_train))

# semantic
inputs = Input(shape=(mydata.MAX_TEXT_LEN,), dtype='int32')
semantic_input = inputs
embedding_matrix = get_embedding_matrix()
embedding = Embedding(len(tk.word_index) + 1,
                      mydata.WORD_VEC_DIM,
                      weights=[embedding_matrix],
                      input_length=mydata.MAX_TEXT_LEN,
                      trainable=False)(inputs)
print(embedding)
print(embedding._keras_shape)

reshape = Reshape((mydata.MAX_TEXT_LEN, mydata.WORD_VEC_DIM, 1))(embedding) # shape:(?,10,5,1)
filter_sizes = (3, 8)
num_filters = 7
conv_0 = Conv2D(filters=num_filters, kernel_size=(filter_sizes[0], mydata.WORD_VEC_DIM), padding='valid',
                activation='relu')(reshape) # shape=(?,8,1,7)
conv_1 = Conv2D(filters=num_filters, kernel_size=(filter_sizes[1], mydata.WORD_VEC_DIM), padding='valid',
                activation='relu')(reshape)# shape=(?,3,1,7)

maxpool_0 = MaxPooling2D(pool_size=(mydata.MAX_TEXT_LEN - filter_sizes[0] + 1, 1), strides=(1, 1), padding='valid')(
    conv_0) # shape=(?,1,1,7)
maxpool_1 = MaxPooling2D(pool_size=(mydata.MAX_TEXT_LEN - filter_sizes[1] + 1, 1), strides=(1, 1), padding='valid')(
    conv_1) # shape=(?,1,1,7)

merged_tensor = concatenate([maxpool_0, maxpool_1], axis=1) # shape=(?,2,1,7)

flatten = Flatten()(merged_tensor)  # shape=(?,?)
# reshape = Reshape((3*num_filters,))(merged_tensor)
dropout = Dropout(0.8)(flatten)
outputs = Dense(units=1)(dropout)
semantic_model = Model(inputs=inputs, outputs=outputs)
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
semantic_model.compile(loss='mean_squared_error',
                       optimizer=adam,
                       metrics=['mean_absolute_error'])
plot_model(semantic_model, to_file='model.png',show_shapes='True')
semantic_model.fit(X_train, y_train, batch_size=5, epochs=20, verbose=1)  # starts training

# numerical
# numerical_model = Sequential()
# numerical_model.add(Dense(100, input_dim=mydata.NUMERICAL_FEATURE_DIM))
# numerical_model.add(BatchNormalization())
# numerical_model.add(PReLU())
# numerical_model.add(Dense(50))

# numerical_model.add(BatchNormalization())
# numerical_model.add(PReLU())
#
# model = Model(input=inputs, output=outputs)
