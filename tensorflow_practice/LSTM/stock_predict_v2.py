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
TRAIN_SIZE = 600
learning_rate = 0.0006  # 学习率


# ——————————————————导入数据——————————————————————



# 获取训练集预测试集
def get_train_test_data():
    file_path = 'd:/in/600036.csv'
    df = pd.read_csv(file_path)  # 读入股票数据
    data = df.values

    # 对各个特征作标准化
    normalized_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    batch_index = []
    data_train = normalized_data[0:TRAIN_SIZE]
    data_test = normalized_data[TRAIN_SIZE:]

    # 训练集
    train_x, train_y = [], []
    for i in range(len(data_train) - TIME_STEP):
        x = data_train[i:i + TIME_STEP, :-1]
        y = data_train[i:i + TIME_STEP, FEATURE_SIZE, np.newaxis]
        train_x.append(x)
        train_y.append(y)

    # 测试集
    test_x, test_y = [], []
    for i in range(len(data_test) - TIME_STEP):
        x = data_test[i:i + TIME_STEP, :-1]
        y = data_test[i:i + TIME_STEP, FEATURE_SIZE, np.newaxis]
        test_x.append(x)
        test_y.append(y)

    return train_x, train_y, test_x, test_y


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
def lstm(features,labels):
    w_in = weights['in']
    b_in = biases['in']
    input = tf.reshape(features, [-1, INPUT_SIZE])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
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


train_x, train_y,test_x, test_y=get_train_test_data()

# ——————————————————训练模型——————————————————
def train_lstm():
    X = tf.placeholder(tf.float32, shape=[None, TIME_STEP, INPUT_SIZE])
    Y = tf.placeholder(tf.float32, shape=[None, TIME_STEP, OUTPUT_SIZE])
    pred, _ = lstm(features=X,labels=Y)
    # 损失函数
    loss =tf.losses.mean_squared_error(labels=Y,predictions=pred)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
    # module_file = tf.train.latest_checkpoint()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, module_file)
        # 重复训练10000次
        for i in range(200):
            _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x, Y: train_y})
            print(i, loss_)
            if i % 200 == 0:
                print("保存模型：", saver.save(sess, './target/stock.model', global_step=i))




# ————————————————预测模型————————————————————
# def prediction(TIME_STEP=15):
#     X = tf.placeholder(tf.float32, shape=[None, TIME_STEP, INPUT_SIZE])
#     # Y=tf.placeholder(tf.float32, shape=[None,TIME_STEP,output_size])
#     mean, std, test_x, test_y = get_test_data(TIME_STEP)
#     pred, _ = lstm(X)
#     saver = tf.train.Saver(tf.global_variables())
#     with tf.Session() as sess:
#         # 参数恢复
#         # module_file = tf.train.latest_checkpoint()
#         module_file = './target/stock.model-1800'
#         saver.restore(sess, module_file)
#         test_predict = []
#         for step in range(len(test_x) - 1):
#             prob = sess.run(pred, feed_dict={X: [test_x[step]]})
#             predict = prob.reshape((-1))
#             test_predict.extend(predict)
#         test_y = np.array(test_y) * std[14] + mean[14]
#         test_predict = np.array(test_predict) * std[14] + mean[14]
#         acc = np.average(np.abs(test_predict - test_y[:len(test_predict)]) / test_y[:len(test_predict)])  # 偏差
#         # 以折线图表示结果
#         plt.figure()
#         plt.plot(list(range(len(test_predict))), test_predict, color='b')
#         plt.plot(list(range(len(test_y))), test_y, color='r')
#         plt.show()




if(__name__=='__main__'):
    print('hi')
    train_lstm()
    # prediction()
