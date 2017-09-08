# 下面的tf程序用来搭建 playground 中的网络结构, 并用同样的样本分布训练它
# http://playground.tensorflow.org/
import tensorflow as tf

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

N_SAMPLES = 1000
# N = N_SAMPLES*(1-0.2)
N = None
TEST_SIZE_PERCENT = 0.2
INPUT_DIMENSION = 2
LEARNING_RATE = 0.03
LOGDIR = 'd:/ttt/pg/'


def get_trainset_and_testset():
    X, y = make_circles(n_samples=N_SAMPLES, noise=0.2, factor=0.2, random_state=1)
    y = y.reshape(N_SAMPLES, 1)
    for i in range(len(y)):
        if y[i][0] == 0:
            y[i][0] = -1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    return X_train, X_test, y_train, y_test


def tf_boy(X_train, y_train):
    x = tf.placeholder(tf.float32, shape=(N, INPUT_DIMENSION), name="x-input")
    y = tf.placeholder(tf.float32, shape=(N, 1), name='y-output')

    w_12 = tf.Variable(tf.random_normal([INPUT_DIMENSION, 4]), name='w_12')
    tf.get_variable()
    w_23 = tf.Variable(tf.random_normal([4, 2]), name='w_23')
    w_34 = tf.Variable(tf.random_normal([2, 1]), name='w_34')

    bias_2 = tf.Variable(tf.zeros([4]), name='bias_2')
    bias_3 = tf.Variable(tf.zeros([2]), name='bias_3')

    out_2 = tf.tanh(tf.matmul(x, w_12) + bias_2)
    out_3 = tf.tanh(tf.matmul(out_2, w_23) + bias_3)
    y_pred = tf.tanh(tf.matmul(out_3, w_34))

    loss = tf.losses.mean_squared_error(y_pred, y_train)

    tf.summary.scalar('loss_value', loss)
    summ = tf.summary.merge_all()

    train = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        writer = tf.summary.FileWriter(LOGDIR, sess.graph)

        # 训练模型
        STEPS = 1000
        loss_arr = []
        for step in range(STEPS):
            sess.run(train, feed_dict={x: X_train, y: y_train})
            # print("w_12:", sess.run(w_12))
            [total_loss, s] = sess.run([loss, summ], feed_dict={x: X_train, y: y_train})
            writer.add_summary(s, step)
            loss_arr.append(total_loss)
            if step % 100 == 0:
                saver.save(sess, LOGDIR + 'model.ckpt', step)

        writer.close()
        print("loss_arr: {} -> {}".format(loss_arr[:5], loss_arr[-5:]))


X_train, X_test, y_train, y_test = get_trainset_and_testset()
tf_boy(X_train, y_train)

"""

"""
