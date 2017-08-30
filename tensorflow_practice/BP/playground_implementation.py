# 下面的tf程序用来搭建 playground 中的网络结构, 并用同样的样本分布训练它
# http://playground.tensorflow.org/
import tensorflow as tf
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

N_SAMPLES=1000
N = N_SAMPLES*(1-0.2)
TEST_SIZE_PERCENT=0.2
INPUT_DIMENSION = 2
LEARNING_RATE = 0.03


def get_trainset_and_testset():
    X, y = make_circles(n_samples=N_SAMPLES, noise=0.2, factor=0.2, random_state=1)
    y=y.reshape(N_SAMPLES,1)
    for i in range(len(y)):
        if y[i][0] == 0:
            y[i][0] = -1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    return X_train, X_test, y_train, y_test


def plot_samples(X, y):
    x1_axis_of_positive_sample = []
    x2_axis_of_positive_sample = []
    x1_axis_of_negative_sample = []
    x2_axis_of_negative_sample = []
    for i in range(len(y)):
        if y[i][0] == 1:
            x1_axis_of_positive_sample.append(X[i][0])
            x2_axis_of_positive_sample.append(X[i][1])
        else:
            x1_axis_of_negative_sample.append(X[i][0])
            x2_axis_of_negative_sample.append(X[i][1])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    type1 = ax.scatter(x1_axis_of_positive_sample, x2_axis_of_positive_sample, c='blue')
    type2 = ax.scatter(x1_axis_of_negative_sample, x2_axis_of_negative_sample, c='brown')
    plt.show()


def tf_boy(X_train, y_train):
    x = tf.placeholder(tf.float32, shape=(N, INPUT_DIMENSION), name="x-input")
    y = tf.placeholder(tf.float32, shape=(N, 1), name='y-output')

    w_12 = tf.Variable(tf.random_normal([INPUT_DIMENSION, 4]))
    w_23 = tf.Variable(tf.random_normal([4, 2]))
    w_34 = tf.Variable(tf.random_normal([2, 1]))

    bias_2 = tf.Variable(tf.zeros([4]))
    bias_3 = tf.Variable(tf.zeros([2]))

    out_2 = tf.tanh(tf.matmul(x, w_12)+bias_2)
    out_3 = tf.tanh(tf.matmul(out_2, w_23)+bias_3)
    y_pred = tf.tanh(tf.matmul(out_3, w_34))

    loss = tf.losses.mean_squared_error(y_pred, y_train)
    train = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # 输出目前（未经训练）的参数取值
        print("w_12:", sess.run(w_12))
        print("bias_2:", sess.run(bias_2))
        print("w_23:", sess.run(w_23))
        print("bias_3:", sess.run(bias_3))
        print("w_34:", sess.run(w_34))
        print("\n")

        # 训练模型
        STEPS = 1000
        loss_arr=[]
        for i in range(STEPS):
            sess.run(train, feed_dict={x: X_train, y: y_train})
            # print("w_12:", sess.run(w_12))
            total_loss = sess.run(loss, feed_dict={x: X_train, y: y_train})
            loss_arr.append(total_loss)

        # 输出训练后的参数取值。
        print("\n")
        print("w_12:", sess.run(w_12))
        print("bias_2:", sess.run(bias_2))
        print("w_23:", sess.run(w_23))
        print("bias_3:", sess.run(bias_3))
        print("w_34:", sess.run(w_34))

        #损失函数值, 变化曲线
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x1_axis=range(STEPS)
        x2_axis=loss_arr
        ax.scatter(x1_axis, x2_axis, c='blue')
        plt.show()





X_train, X_test, y_train, y_test = get_trainset_and_testset()
plot_samples(X_train, y_train)
tf_boy(X_train, y_train)

"""
w_12: [[-1.02140009  0.1605026  -1.79282069  0.4733575 ]
 [ 0.39666429  0.18492372 -0.76060712  1.30654573]]
bias_2: [ 0.  0.  0.  0.]
w_23: [[ 0.67617249  0.37842372]
 [-0.13533577 -0.66545761]
 [-1.15543711 -2.49448705]
 [ 2.22956133 -0.32165667]]
bias_3: [ 0.  0.]
w_34: [[-0.3632547 ]
 [ 1.46542287]]




w_12: [[-1.49568939  0.37617555 -1.63048255  0.28560442]
 [ 0.9723621   0.18761159 -0.50179791  1.75902331]]
bias_2: [ 0.82246989  0.85552055 -0.75419569 -1.0005064 ]
w_23: [[ 0.47892851  1.46363699]
 [-0.24798357 -1.31343782]
 [-0.90114665 -1.79665875]
 [ 2.36001635 -1.09818363]]
bias_3: [-0.33126104 -1.38699543]
w_34: [[-0.48449364]
 [ 2.55953455]]

"""