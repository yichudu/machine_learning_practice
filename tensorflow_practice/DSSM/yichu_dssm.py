import tensorflow as tf
import numpy as np

tf.app.flags.DEFINE_integer('mock_dataset_size', 5, 'mock 时数据集的大小')

tf.app.flags.DEFINE_integer('max_text_len', 200, 'max text len')
tf.app.flags.DEFINE_integer('max_vocab_size', int(6E5), 'vocabulary size')
tf.app.flags.DEFINE_integer('word_vec_dim', 100, 'word_vec_dim')

tf.app.flags.DEFINE_integer('max_tag_size', int(1E5), 'tag size')
tf.app.flags.DEFINE_integer('tag_negative_number', 3, 'tag_negative_number')
tf.app.flags.DEFINE_integer('tag_vec_dim', 100, 'tag_vec_dim')

tf.app.flags.DEFINE_integer('project_space_dim', 128, 'project_space_dim')
tf.app.flags.DEFINE_integer('batch_size', 10, 'batch size')

tf.app.flags.DEFINE_float('learning_rate', 0.03, 'batch size')


FLAGS = tf.app.flags.FLAGS


def train_input_fn():
    # mock
    content_sequence = np.random.randint(1, FLAGS.max_vocab_size, size=[FLAGS.mock_dataset_size, FLAGS.max_text_len])
    positive_tag = np.random.randint(1, FLAGS.max_tag_size, size=[FLAGS.mock_dataset_size, 1])
    negative_tag = np.random.randint(1, FLAGS.max_tag_size, size=[FLAGS.mock_dataset_size, FLAGS.tag_negative_number])
    my_data = np.hstack((content_sequence, positive_tag, negative_tag))
    dataset = tf.data.Dataset.from_tensor_slices(my_data)
    dataset = dataset.shuffle(1000).repeat(100).batch(FLAGS.batch_size)
    next_item = dataset.make_one_shot_iterator().get_next()

    # Build the Iterator, and return the read end of the pipeline.
    return next_item


content_input = tf.placeholder(tf.int32, shape=(None, FLAGS.max_text_len))
content_embedding_matrix = tf.get_variable(name='content_embedding_matrix',
                                           shape=[FLAGS.max_vocab_size, FLAGS.word_vec_dim])
content_embedding = tf.nn.embedding_lookup(content_embedding_matrix, content_input)
content_embedding = tf.reshape(content_embedding, shape=[-1, FLAGS.max_text_len, FLAGS.word_vec_dim, 1])
content_conv = tf.layers.Conv2D(filters=100, kernel_size=[3, FLAGS.word_vec_dim])

content_conv = content_conv(content_embedding)
content_max_pooling = tf.layers.max_pooling2d(content_conv, pool_size=(FLAGS.max_text_len - 3 + 1, 1), strides=(1, 1))
content_flatten = tf.layers.flatten(content_max_pooling, name='content_flatten')
content_project = tf.contrib.layers.fully_connected(content_flatten, FLAGS.project_space_dim, tf.nn.relu)

# tag_positive_input = tf.layers.Input(shape=(1,))
# tag_negative_inputs =[ tf.layers.Input(shape=(1,)) for i in range(3)]
tag_positive_input = tf.placeholder(tf.int32, shape=(None, 1))
#tag_negative_input_arr = [tf.placeholder(tf.int32, shape=(None, 1)) for i in range(FLAGS.tag_negative_number)]
tag_negative_input_0 = tf.placeholder(tf.int32, shape=(None, 1))
tag_negative_input_1 = tf.placeholder(tf.int32, shape=(None, 1))
tag_negative_input_2 = tf.placeholder(tf.int32, shape=(None, 1))
tag_negative_input_arr=[tag_negative_input_0,tag_negative_input_1,tag_negative_input_2]

tag_embedding_matrix = tf.get_variable(name='tag_embedding_matrix', shape=[FLAGS.max_tag_size, FLAGS.tag_vec_dim])
tag_embedding_positive = tf.nn.embedding_lookup(tag_embedding_matrix, tag_positive_input)
tag_embedding_negative_arr = [tf.nn.embedding_lookup(tag_embedding_matrix, tag_negative_input) for tag_negative_input in
                              tag_negative_input_arr]

tag_embedding_positive = tf.reshape(tag_embedding_positive, shape=[-1, FLAGS.tag_vec_dim])
tag_embedding_negative_arr = [tf.reshape(tag_embedding_negative, shape=[-1, FLAGS.tag_vec_dim]) for
                              tag_embedding_negative in tag_embedding_negative_arr]

dense_layer = tf.layers.Dense(FLAGS.project_space_dim, activation=tf.nn.relu)

tag_positive_project = dense_layer(tag_embedding_positive)
tag_negative_project_arr = [dense_layer(tag_embedding_negative) for tag_embedding_negative in
                            tag_embedding_negative_arr]

#  a dot product between two tensors
positive_dot = tf.reduce_sum(tf.multiply( content_project, tag_positive_project), axis=1)
negative_dot_arr = [tf.reduce_sum(tf.multiply( content_project, tag_negative_project), axis=1)for tag_negative_project in
                    tag_negative_project_arr]

positive_dot=tf.reshape(positive_dot,shape=(-1,1))
negative_dot_arr=[ tf.reshape(negative_dot,shape=(-1,1)) for negative_dot in negative_dot_arr]
concate_output = tf.concat([positive_dot] + negative_dot_arr, axis=1)
# arg`labels` is from `0` on
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=[0]*FLAGS.batch_size, logits=concate_output)
train = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss)

init_op = tf.global_variables_initializer()


def train_fn():
    with tf.Session() as sess:
        sess.run(init_op)
        train_data_itrator = train_input_fn()
        i=0
        while True:
            try:
                data_read = sess.run(train_data_itrator)
                _, loss_evl = sess.run([train, loss], feed_dict=
                {content_input: data_read[:,:FLAGS.max_text_len],
                 tag_positive_input: data_read[:,FLAGS.max_text_len:-3],
                 tag_negative_input_0: data_read[:,-3:-2],
                 tag_negative_input_1: data_read[:, -2:-1],
                 tag_negative_input_2: data_read[:, -1:],
                 })
                print(np.mean(loss_evl))
                i+=1

            except tf.errors.OutOfRangeError:
                print('data has been read up')
                print(i)
                break
            except Exception as ex:
                print(ex)
                break



if __name__ == '__main__':
    train_fn()
