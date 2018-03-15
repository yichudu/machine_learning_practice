import tensorflow as tf
import numpy as np

tf.app.flags.DEFINE_integer('mock_dataset_size', 5, 'mock 时数据集的大小')

tf.app.flags.DEFINE_integer('max_text_len', 10, 'max text len')
tf.app.flags.DEFINE_integer('max_vocab_size', int(6E5), 'vocabulary size')
tf.app.flags.DEFINE_integer('word_vec_dim', 100, 'word_vec_dim')

tf.app.flags.DEFINE_integer('max_tag_size', int(1E5), 'tag size')
tf.app.flags.DEFINE_integer('tag_negative_number', 3, 'tag_negative_number')
tf.app.flags.DEFINE_integer('tag_vec_dim', 100, 'tag_vec_dim')

tf.app.flags.DEFINE_integer('project_space_dim', 128, 'project_space_dim')
tf.app.flags.DEFINE_integer('batch_size', 2, 'batch size')
tf.app.flags.DEFINE_integer('train_steps', 100, 'train_steps')

tf.app.flags.DEFINE_float('learning_rate', 0.03, 'batch size')

FLAGS = tf.app.flags.FLAGS

MAX_TEXT_LEN = FLAGS.max_text_len
TAG_NEGATIVE_NUMBER = FLAGS.tag_negative_number


def train_input_fn(features=None, labels=None, batch_size=100):
    # mock
    if features is None:
        content_sequence = np.random.randint(1, FLAGS.max_vocab_size, size=[FLAGS.mock_dataset_size, MAX_TEXT_LEN])
        pos_tag_id = np.random.randint(1, FLAGS.max_tag_size, size=[FLAGS.mock_dataset_size, 1])
        neg_tag_id_arr = np.random.randint(1, FLAGS.max_tag_size, size=[FLAGS.mock_dataset_size, TAG_NEGATIVE_NUMBER])
        features = {'content_sequence': content_sequence,
                   'pos_tag_id': pos_tag_id,
                   'neg_tag_id_arr': neg_tag_id_arr
                   }
        labels=np.array([0] * FLAGS.mock_dataset_size)
    # my_data = np.hstack((content_sequence, positive_tag, negative_tag))
    dataset = tf.data.Dataset.from_tensor_slices((features,labels))
    dataset = dataset.shuffle(1000).repeat(100).batch(batch_size)

    return dataset
    next_item = dataset.make_one_shot_iterator().get_next()
    # Build the Iterator, and return the read end of the pipeline.
    # return next_item


def get_feature_columns():
    # Feature columns describe how to use the input.
    my_feature_columns = []
    my_feature_columns.append(tf.feature_column.numeric_column(key='content_sequence', shape=(MAX_TEXT_LEN,),dtype=tf.int32))
    my_feature_columns.append(tf.feature_column.numeric_column(key='pos_tag_id',dtype=tf.int32))
    my_feature_columns.append(tf.feature_column.numeric_column(key='neg_tag_id_arr', shape=(TAG_NEGATIVE_NUMBER,)))
    return my_feature_columns


def yichu_dssm_model_fn(
        features,  # This is batch_features from input_fn
        labels,  # This is batch_labels from input_fn
        mode,  # An instance of tf.estimator.ModeKeys
        params):
    if mode == tf.estimator.ModeKeys.PREDICT:
        pass

    input_layer = tf.feature_column.input_layer(features, params['feature_columns'])
    input_layer=tf.cast(input_layer,dtype=tf.int32)
    content_input, tag_positive_input, tag_negative_input_0,tag_negative_input_1,tag_negative_input_2 = tf.split(input_layer,
                                                                         [MAX_TEXT_LEN, 1]+[1]*TAG_NEGATIVE_NUMBER, 1)
    tag_negative_input_arr=[tag_negative_input_0,tag_negative_input_1,tag_negative_input_2]
    content_embedding_matrix = tf.get_variable(name='content_embedding_matrix',
                                               shape=[FLAGS.max_vocab_size, FLAGS.word_vec_dim])
    content_embedding = tf.nn.embedding_lookup(content_embedding_matrix, content_input)
    content_embedding = tf.reshape(content_embedding, shape=[-1, MAX_TEXT_LEN, FLAGS.word_vec_dim, 1])
    content_conv = tf.layers.Conv2D(filters=100, kernel_size=[3, FLAGS.word_vec_dim])

    content_conv = content_conv(content_embedding)
    content_max_pooling = tf.layers.max_pooling2d(content_conv, pool_size=(MAX_TEXT_LEN - 3 + 1, 1),
                                                  strides=(1, 1))
    content_flatten = tf.layers.flatten(content_max_pooling, name='content_flatten')
    content_project = tf.contrib.layers.fully_connected(content_flatten, FLAGS.project_space_dim, tf.nn.relu)

    tag_embedding_matrix = tf.get_variable(name='tag_embedding_matrix', shape=[FLAGS.max_tag_size, FLAGS.tag_vec_dim])
    tag_embedding_positive = tf.nn.embedding_lookup(tag_embedding_matrix, tag_positive_input)
    tag_embedding_negative_arr = [tf.nn.embedding_lookup(tag_embedding_matrix, tag_negative_input) for
                                  tag_negative_input in
                                  tag_negative_input_arr]

    tag_embedding_positive = tf.reshape(tag_embedding_positive, shape=[-1, FLAGS.tag_vec_dim])
    tag_embedding_negative_arr = [tf.reshape(tag_embedding_negative, shape=[-1, FLAGS.tag_vec_dim]) for
                                  tag_embedding_negative in tag_embedding_negative_arr]

    dense_layer = tf.layers.Dense(FLAGS.project_space_dim, activation=tf.nn.relu)

    tag_positive_project = dense_layer(tag_embedding_positive)
    tag_negative_project_arr = [dense_layer(tag_embedding_negative) for tag_embedding_negative in
                                tag_embedding_negative_arr]

    #  a dot product between two tensors
    positive_dot = tf.reduce_sum(tf.multiply(content_project, tag_positive_project), axis=1)
    negative_dot_arr = [tf.reduce_sum(tf.multiply(content_project, tag_negative_project), axis=1) for
                        tag_negative_project in
                        tag_negative_project_arr]

    positive_dot = tf.reshape(positive_dot, shape=(-1, 1))
    negative_dot_arr = [tf.reshape(negative_dot, shape=(-1, 1)) for negative_dot in negative_dot_arr]
    concate_output = tf.concat([positive_dot] + negative_dot_arr, axis=1)
    # arg`labels` is from `0` on
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=concate_output)
    train_op = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def train_fn():
    classifier = tf.estimator.Estimator(
        model_fn=yichu_dssm_model_fn,
        params={'feature_columns': get_feature_columns()})
    classifier.train(input_fn=lambda :train_input_fn(batch_size=FLAGS.batch_size), steps=FLAGS.train_steps)


if __name__ == '__main__':
    train_fn()
