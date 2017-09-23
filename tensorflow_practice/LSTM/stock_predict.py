# coding=utf-8
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 定义常量
TIME_STEP = 20
RNN_UNIT = 10  # hidden layer units
INPUT_SIZE = 14
FEATURE_SIZE = INPUT_SIZE
OUTPUT_SIZE = 1
BATCH_SIZE = 60
lr = 0.0006  # 学习率
# ——————————————————导入数据——————————————————————
f = open('600036.csv')
# data=csv.reader(open('600036.csv'))
df = pd.read_csv(f)  # 读入股票数据
data = df.iloc[:, 0:15].values


# 获取训练集
def get_train_data(batch_size=BATCH_SIZE, time_step=TIME_STEP, train_begin=0, train_end=600):
    batch_index = []
    data_train = data[train_begin:train_end]
    normalized_train_data = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)  # 对各个特征作标准化
    train_x, train_y = [], []  # 训练集
    for i in range(len(normalized_train_data) - time_step):
        if i % batch_size == 0:
            batch_index.append(i)
        x = normalized_train_data[i:i + time_step, :FEATURE_SIZE]
        y = normalized_train_data[i:i + time_step, FEATURE_SIZE, np.newaxis]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data) - time_step))
    return batch_index, train_x, train_y


# 获取测试集
def get_test_data(time_step=20, test_begin=600):
    data_test = data[test_begin:]
    mean = np.mean(data_test, axis=0)
    std = np.std(data_test, axis=0)
    normalized_test_data = (data_test - mean) / std  # 标准化
    size = (len(normalized_test_data) + time_step - 1) // time_step  # 有size个sample
    test_x, test_y = [], []
    for i in range(size - 1):
        x = normalized_test_data[i * time_step:(i + 1) * time_step, :FEATURE_SIZE]
        y = normalized_test_data[i * time_step:(i + 1) * time_step, FEATURE_SIZE]
        test_x.append(x.tolist())
        test_y.extend(y)
    test_x.append((normalized_test_data[(i + 1) * time_step:, :FEATURE_SIZE]).tolist())
    test_y.extend((normalized_test_data[(i + 1) * time_step:, FEATURE_SIZE]).tolist())
    return mean, std, test_x, test_y


# ——————————————————定义神经网络变量——————————————————
# 输入层、输出层权重、偏置

weights = {
    'in': tf.Variable(tf.random_normal([INPUT_SIZE, RNN_UNIT])),
    'out': tf.Variable(tf.random_normal([RNN_UNIT, 1]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[RNN_UNIT, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
}


# ——————————————————定义神经网络变量——————————————————
def lstm(X):
    """


    :param X:
        X = tf.placeholder(tf.float32, shape=[None, time_step, INPUT_SIZE])
    :return:
    """
    w_in = weights['in']
    b_in = biases['in']
    input = tf.reshape(X, [-1, INPUT_SIZE])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, TIME_STEP, RNN_UNIT])  # 将tensor转成3维，作为lstm cell的输入
    cell = tf.nn.rnn_cell.BasicLSTMCell(RNN_UNIT)
    init_state = cell.zero_state(BATCH_SIZE, dtype=tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state,
                                                 dtype=tf.float32)  # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output = tf.reshape(output_rnn, [-1, RNN_UNIT])  # 作为输出层的输入
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out) + b_out
    return pred, final_states


# ——————————————————训练模型——————————————————
def train_lstm(batch_size=BATCH_SIZE, time_step=TIME_STEP, train_begin=0, train_end=600):
    X = tf.placeholder(tf.float32, shape=[None, time_step, INPUT_SIZE])
    Y = tf.placeholder(tf.float32, shape=[None, time_step, OUTPUT_SIZE])
    batch_index, train_x, train_y = get_train_data(batch_size, time_step, train_begin, train_end)
    pred, _ = lstm(X)
    # 损失函数
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
    # module_file = tf.train.latest_checkpoint()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, module_file)
        # 重复训练10000次
        for i in range(2000):
            for step in range(len(batch_index) - 1):
                _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],
                                                                 Y: train_y[batch_index[step]:batch_index[step + 1]]})
            print(i, loss_)
            if i % 200 == 0:
                print("保存模型：", saver.save(sess, 'stock.model', global_step=i))


train_lstm()


# ————————————————预测模型————————————————————
def prediction(time_step=15):
    X = tf.placeholder(tf.float32, shape=[None, time_step, INPUT_SIZE])
    # Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    mean, std, test_x, test_y = get_test_data(time_step)
    pred, _ = lstm(X)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 参数恢复
        # module_file = tf.train.latest_checkpoint()
        module_file = './target/stock.model-1800'
        saver.restore(sess, module_file)
        test_predict = []
        for step in range(len(test_x) - 1):
            prob = sess.run(pred, feed_dict={X: [test_x[step]]})
            predict = prob.reshape((-1))
            test_predict.extend(predict)
        test_y = np.array(test_y) * std[14] + mean[14]
        test_predict = np.array(test_predict) * std[14] + mean[14]
        acc = np.average(np.abs(test_predict - test_y[:len(test_predict)]) / test_y[:len(test_predict)])  # 偏差
        # 以折线图表示结果
        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='b')
        plt.plot(list(range(len(test_y))), test_y, color='r')
        plt.show()


prediction()
