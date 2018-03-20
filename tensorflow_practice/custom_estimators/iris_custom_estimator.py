"""
与官方教程的区别是:
 1. 数据源不是pandas的DataFrame, 而是 numpy 的 ndarray
 2. 四个数值特征合并为一个 shape=(4,) 的特征.
"""

import tensorflow as tf
from sklearn import datasets
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)


def train_input_fn(features=None, labels=None, batch_size=100):
    if features is None:
        iris = datasets.load_iris()
        features = {'numerical_features': iris.data}  # ndarray, shape is (150,4), dtype is float64
        labels = iris.target  # ndarray, shape is (150,), dtype is int32, range is [0,2]
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(1000).repeat(100).batch(batch_size)

    return dataset


def my_model(features, labels, mode, params):
    """DNN with three hidden layers, and dropout of 0.1 probability."""
    # Create three fully connected layers each layer having a dropout
    # probability of 0.1.
    # print(features)
    net = tf.feature_column.input_layer(features, params['feature_columns']) #Tensor("input_layer/concat:0", shape=(?, 4), dtype=float32)
    # print(net)
    # print(tf.Print(net,[net],message='yyyyyyyyyyyyyyyyyyyyyy'))

    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)

    # Compute loss.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def train():
    feature_columns = [tf.feature_column.numeric_column(key='numerical_features', shape=4)]
    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        model_dir='d:/tf_models/iris',
        params={
            'feature_columns': feature_columns,
            # Two hidden layers of 10 nodes each.
            'hidden_units': [5, 10],
            # The model must choose between 3 classes.
            'n_classes': 3,
        })

    # Train the Model.
    classifier.train(
        input_fn=lambda: train_input_fn(None, None, batch_size=7),
        steps=100)


train()
