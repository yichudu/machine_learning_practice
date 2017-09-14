# tf.__version__  1.3.0
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat
from tensorflow.contrib.layers import fully_connected
import matplotlib.pyplot as plt

learn = tf.contrib.learn
import time

NUM_UNITS = 10  # MUST be equal to TIME_STEPS
NUM_LAYERS = 2

TIME_STEPS = 10
# 只有一个特征, 即数值本身
INPUT_SIZE=1
TRAINING_STEPS = 1000
BATCH_SIZE = 32

TRAINING_EXAMPLES = 10000
TESTING_EXAMPLES = 1000
SAMPLE_GAP = 0.01


# X.shape=(9989,1,10)   Y.shape=(9989,1)
# sample is [(x1, ... , x10),y], namely [(v_{t-10}, ... , v_{t-1}),v_t]
def generate_data(seq):
    X = []
    y = []

    for i in range(len(seq) - TIME_STEPS - 1):
        X.append([seq[i: i + TIME_STEPS]])
        y.append([seq[i + TIME_STEPS]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def lstm_model(features, labels):
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(NUM_UNITS, state_is_tuple=True)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * NUM_LAYERS)

    output, _ = tf.nn.dynamic_rnn(cell, features, dtype=tf.float32)
    output = tf.reshape(output, [-1, NUM_UNITS])

    # 创建一个全连接层，因为是回归问题, 输出的维度为1，None指的是不使用激活函数
    predictions = tf.contrib.layers.fully_connected(output, 1, None)

    # 将predictions和labels调整统一的shape
    labels = tf.reshape(labels, [-1])
    predictions = tf.reshape(predictions, [-1])

    loss = tf.losses.mean_squared_error(predictions, labels)

    train_op = tf.contrib.layers.optimize_loss(
        loss, tf.contrib.framework.get_global_step(),
        optimizer="Adagrad", learning_rate=0.1)

    return predictions, loss, train_op


timestamp = time.clock()
# 生成数据
test_start = TRAINING_EXAMPLES * SAMPLE_GAP
test_end = (TRAINING_EXAMPLES + TESTING_EXAMPLES) * SAMPLE_GAP
train_X, train_y = generate_data(np.sin(np.linspace(
    0, test_start, TRAINING_EXAMPLES, dtype=np.float32)))
test_X, test_y = generate_data(np.sin(np.linspace(
    test_start, test_end, TESTING_EXAMPLES, dtype=np.float32)))

# 封装之前定义的lstm。
regressor = SKCompat(learn.Estimator(model_fn=lstm_model, model_dir="../../target/lstm_model/"))
# 训练
regressor.fit(train_X, train_y, batch_size=BATCH_SIZE, steps=TRAINING_STEPS)

# 计算预测值
predicted = [[pred] for pred in regressor.predict(test_X)]

# 计算MSE
mse = np.sqrt(((predicted - test_y) ** 2).mean(axis=0))
print("Mean Square Error is: {}".format(mse[0]))

print("time cost : {} s".format(time.clock() - timestamp))

plot_predicted, = plt.plot(predicted, label='$\hat y$')
plot_test, = plt.plot(test_y, label='y=sin(x)')
plt.legend([plot_predicted, plot_test], ['$\hat y$', 'y=sin(x)'])
plt.show()

"""
Mean Square Error is: 0.00637664133682847
time cost : 11.2319511742214 s

"""
