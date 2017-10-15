# tf.__version__  1.3.0
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat
from tensorflow.contrib.layers import fully_connected
import matplotlib.pyplot as plt

# import matplotlib
# matplotlib.use('Qt5Agg')

from tensorflow_practice.LSTM.sin_x import inference

learn = tf.contrib.learn
import time

NUM_UNITS = 8
NUM_LAYERS = 2

TIME_STEPS = 10
# 只有一个特征, 即数值本身
INPUT_SIZE = 1
OUTPUT_SIZE = 1

TRAINING_STEPS = 1000
BATCH_SIZE = 5

TRAINING_EXAMPLES = 10000
TESTING_EXAMPLES = 1000
SAMPLE_GAP = 0.01

TEST_START = TRAINING_EXAMPLES * SAMPLE_GAP
TEST_END = (TRAINING_EXAMPLES + TESTING_EXAMPLES) * SAMPLE_GAP

MODEL_SAVE_PATH = 'd:/out/tf_models/sin_x/'


# MODEL_NAME='sin_x.ckpt'
def train():
    X = tf.placeholder(tf.float32, shape=[None, TIME_STEPS, INPUT_SIZE])
    Y = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE])
    out_tensor = Y
    predictions = inference.inference(X)
    loss = tf.losses.mean_squared_error(predictions, out_tensor)

    train_op = tf.contrib.layers.optimize_loss(
        loss, tf.contrib.framework.get_global_step(),
        optimizer="Adagrad", learning_rate=0.1)

    saver = tf.train.Saver()

    train_X, train_y = inference.generate_data(condition='train')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loss_result_arr = []
        global_step = 0
        steps = int(len(train_X) / BATCH_SIZE)
        for epoch in range(1):
            # iteration
            for step in range(steps):
                batch_X = train_X[step * BATCH_SIZE: (step + 1) * BATCH_SIZE]
                batch_y = train_y[step * BATCH_SIZE: (step + 1) * BATCH_SIZE]
                loss_result, _ = sess.run([loss, train_op], feed_dict={X: batch_X, Y: batch_y})
                loss_result_arr.append(loss_result)
                global_step += 1

            saver.save(sess, MODEL_SAVE_PATH, global_step)


# entry
train()
