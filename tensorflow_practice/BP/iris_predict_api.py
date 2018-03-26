"""
鸢尾花,三层三分类的神经网络
"""
import tensorflow as tf

from sklearn import datasets
from sklearn import model_selection

import numpy as np

INPUT_DIMENSION = 4



def train_input_fn():
    iris = datasets.load_iris()
    X_train, X_test, y_train, y_test = model_selection.train_test_split(iris.data, iris.target, test_size=0.2)
    # y_train = np.reshape(y_train, newshape=(-1, 1))
    features=X_train
    labels=y_train
    return features,labels

def get_placeholder():
    features = tf.placeholder(tf.float32, shape=(None, INPUT_DIMENSION), name="features_tensor")
    labels = tf.placeholder(tf.int32, shape=(None, ), name='labels_tensor')
    return features,labels

def model_fn(features,  # This is batch_features from input_fn
        labels,  # This is batch_labels from input_fn
        mode,  # An instance of tf.estimator.ModeKeys
        params=None):
    """

    :param hidden_units: tuple
    :return: train_op, loss_tensor,predict_tensor
    """
    net=features
    hidden_units=[10,10]
    for units in hidden_units:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # Compute logits (1 per class).
    logits = tf.layers.dense(net, 3, activation=tf.nn.relu)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)

    # Compute loss.
    loss_tensor = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    predict_tensor = tf.argmax(logits, 1,name='predict_tensor')
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss_tensor, global_step=tf.train.get_global_step())

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Compute predictions.
        return predict_tensor

    if mode == tf.estimator.ModeKeys.TRAIN:
        return train_op, loss_tensor
    if mode is None:
        return train_op, loss_tensor,predict_tensor


features_placeholder, labels_placeholder=get_placeholder()
features,labels=train_input_fn()

def train():
    train_op, loss_tensor=model_fn(features_placeholder, labels_placeholder,mode=tf.estimator.ModeKeys.TRAIN)
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        # 训练模型
        STEPS = 1000
        for i in range(STEPS):
            _,loss_evl = sess.run([train_op, loss_tensor], feed_dict={features_placeholder: features, labels_placeholder: labels})
            if i % 100 == 0:
                print(loss_evl)

if __name__ == '__main__':
    train()

