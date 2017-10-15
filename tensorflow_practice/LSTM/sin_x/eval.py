# tf.__version__  1.3.0
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat
from tensorflow.contrib.layers import fully_connected
import matplotlib.pyplot as plt

from tensorflow_practice.LSTM.sin_x import inference
from tensorflow_practice.LSTM.sin_x import train

test_X, test_y = inference.generate_data(condition='test')

def eval():
    inference.inference()

    variables_to_restore =
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH)
    if ckpt and ckpt.model_checkpoint_state:
        saver.restore
