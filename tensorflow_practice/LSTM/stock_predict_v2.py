# coding=utf-8
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 定义常量
TIME_STEP = 20
OUTPUT_TIME_STEP = 1
NUM_UNITS = 10  # hidden layer units
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
    data_train = normalized_data[0:TRAIN_SIZE]
    data_test = normalized_data[TRAIN_SIZE:]

    # 训练集
    train_x, train_y = [], []
    for i in range(len(data_train) - TIME_STEP):
        x = data_train[i:i + TIME_STEP, :-1]
        y = data_train[i + TIME_STEP - 1, FEATURE_SIZE]
        train_x.append(x)
        train_y.append(y)

    # 测试集
    test_x, test_y = [], []
    for i in range(len(data_test) - TIME_STEP):
        x = data_test[i:i + TIME_STEP, :-1]
        y = data_test[i + TIME_STEP - 1, FEATURE_SIZE]
        test_x.append(x)
        test_y.append(y)

    train_X=np.array(train_x)
    train_y = np.array(train_y)
    test_X = np.array(test_x)
    test_y = np.array(test_y)

    train_X=np.reshape(train_X,[-1,TIME_STEP,INPUT_SIZE])
    train_y =np.reshape(train_X,[-1,OUTPUT_SIZE])
    test_X = np.reshape(train_X,[-1,TIME_STEP,INPUT_SIZE])
    test_y = np.reshape(train_X,[-1,OUTPUT_SIZE])
    return train_X, train_y, test_X, test_y




# ——————————————————定义神经网络变量——————————————————
def lstm(in_tensor, out_tensor):
    w_in = tf.get_variable('weight_input', shape=[INPUT_SIZE, NUM_UNITS], initializer=tf.truncated_normal_initializer())
    # 注意 shape=[NUM_UNITS,] 与 shape=[NUM_UNITS,1] 的严重不同!
    b_in = tf.get_variable('bias_input', shape=[NUM_UNITS, ], initializer=tf.zeros_initializer())

    input = tf.reshape(in_tensor, [-1, INPUT_SIZE])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, TIME_STEP, NUM_UNITS])  # 将tensor转成3维，作为lstm cell的输入
    cell = tf.nn.rnn_cell.BasicLSTMCell(NUM_UNITS)
    init_state = cell.zero_state(BATCH_SIZE, dtype=tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state,
                                                 dtype=tf.float32)
    output_rnn = output_rnn[:, -1, :]
    # 创建一个全连接层，因为是回归问题, 输出的维度为1，None指的是不使用激活函数
    predictions = tf.contrib.layers.fully_connected(output_rnn, 1, None)

    # 将predictions和labels调整统一的shape
    # labels = tf.reshape(labels, [-1])
    predictions = tf.reshape(predictions, [-1, OUTPUT_TIME_STEP])
    loss = tf.losses.mean_squared_error(predictions, out_tensor)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return predictions, loss,train_op


train_X, train_y,test_X, test_y=get_train_test_data()

# ——————————————————训练模型——————————————————
def train_lstm():
    X = tf.placeholder(tf.float32, shape=[None, TIME_STEP, INPUT_SIZE])
    Y = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE])
    pred, loss,train_op = lstm(in_tensor=X, out_tensor=Y)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
    # module_file = tf.train.latest_checkpoint()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        steps = int(len(train_X) / BATCH_SIZE)
        # iteration
        for step in range(steps):
            batch_X = train_X[step * BATCH_SIZE: (step + 1) * BATCH_SIZE]
            batch_y = train_y[step * BATCH_SIZE: (step + 1) * BATCH_SIZE]
        # saver.restore(sess, module_file)
            _, loss_ = sess.run([train_op, loss], feed_dict={X: batch_X, Y: batch_y})
            print(step, loss_)




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
