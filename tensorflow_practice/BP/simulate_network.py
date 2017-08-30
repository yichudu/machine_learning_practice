"""
用tf仿真一个已知的简单神经网络,
网络的图片地址:    http://img.blog.csdn.net/20170816081420914
预期的结果使用ndarray计算, 与tf的输出结果作对比
"""
import tensorflow as tf
import numpy as np

x_mat = [[1, 2]]
w1_mat = np.array([[0, 1], [2, 3]])
w2_mat = np.array([[1], [2]])

x = tf.constant(x_mat)
w1 = tf.Variable(w1_mat)
w2 = tf.Variable(w2_mat)
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)


def calc_tf():
    with tf.Session() as sess:
        sess.run(w1.initializer)
        sess.run(w2.initializer)
        result = sess.run(y)
    return result


def calc_ndarray():
    tmp = np.dot(x_mat, w1_mat)
    tmp = np.dot(tmp, w2_mat)
    return tmp


if (calc_tf() == calc_ndarray()):
    print('They\'re equal, and value is ' + (str)(calc_tf()))
else:
    print('something wrong')

"""
They're equal, and value is [[18]]
"""
