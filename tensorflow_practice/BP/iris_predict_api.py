"""
鸢尾花,三层三分类的神经网络
"""
import tensorflow as tf

from sklearn import datasets
from sklearn import model_selection

import numpy as np

INPUT_DIMENSION = 4
x = tf.placeholder(tf.float32, shape=(None, INPUT_DIMENSION), name="x-input")
y = tf.placeholder(tf.float32, shape=(None, 1), name='y-output')
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = model_selection.train_test_split(iris.data, iris.target, test_size=0.2)
y_train = np.reshape(y_train, newshape=(-1, 1))

w_12 = tf.Variable(tf.random_normal([INPUT_DIMENSION, 4]))
w_23 = tf.Variable(tf.random_normal([4, 3]))

bias_2 = tf.Variable(tf.zeros([4]))
bias_3 = tf.Variable(tf.zeros([3]))

out_2 = tf.nn.leaky_relu(tf.nn.xw_plus_b(x, w_12, bias_2))
y_pred = tf.nn.leaky_relu(tf.nn.xw_plus_b(out_2, w_23, bias_3))

loss = tf.losses.sparse_softmax_cross_entropy(y_train, y_pred)
train = tf.train.AdamOptimizer().minimize(loss)

if __name__ == '__main__':
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # 训练模型
        STEPS = 1000
        loss_arr = []
        for i in range(STEPS):
            loss_evl, _ = sess.run([loss, train], feed_dict={x: X_train, y: y_train})
            if i%100==0:
                print(loss_evl)

