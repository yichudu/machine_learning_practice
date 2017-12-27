import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import PReLU
from keras.preprocessing import sequence, text
from sklearn import preprocessing
from keras.layers.merge import Concatenate
from sklearn.model_selection import train_test_split
import numpy as np
import keras
from keras.layers import Merge
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

DOC_VEC_DIM=10
NUMERICAL_FEATURE_DIM=5
y_train=[0.8,0.9]

# semantic
semantic_input=np.random.standard_normal(size=(2,10))

semantic_model= Sequential()
semantic_model.add(Dense(100,input_dim=DOC_VEC_DIM))
semantic_model.add(PReLU())
semantic_model.add(Dropout(0.2))
semantic_model.add(BatchNormalization())

# numerical
numerical_input=np.random.standard_normal(size=(2,NUMERICAL_FEATURE_DIM))

numerical_model= Sequential()
numerical_model.add(Dense(100, input_dim=NUMERICAL_FEATURE_DIM))
numerical_model.add(PReLU())
numerical_model.add(Dropout(0.2))
numerical_model.add(BatchNormalization())

#merge
merged_model=Sequential()
""" notice the `keras.legacy.layers.Merge` class. not confused with

- `keras.layers.merge.Concatenate(_Merge)` class
- `keras.layers.merge(inputs, mode='sum', concat_axis=-1,
          dot_axes=-1, output_shape=None, output_mask=None,
          arguments=None, name=None)` method
- `keras.layers.merge.concatenate(inputs, axis=-1, **kwargs)` method
"""

merge_layer_1=keras.layers.Merge(layers=[semantic_model,numerical_model],mode='concat')

merged_model.add(merge_layer_1)
merged_model.add(Dense(100, input_dim=numerical_input.shape[1]))
merged_model.add(PReLU())
merged_model.add(Dense(1))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
merged_model.compile(loss='mean_squared_error',
              optimizer=sgd,
              metrics=['mae'])

merged_model.fit(x=[semantic_input,numerical_input], y=y_train)