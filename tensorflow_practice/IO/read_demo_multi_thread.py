import tensorflow as tf
import time
import threading

tf.app.flags.DEFINE_string("tables", "", "tables info")

FLAGS = tf.app.flags.FLAGS

print("tables:" + FLAGS.tables)
tables = [FLAGS.tables]
tables = ['d:/in/a.csv']
NUM_EPOCHS=2


def get_data():
    filename_queue = tf.train.string_input_producer(tables, num_epochs=NUM_EPOCHS)

    # reader = tf.TableRecordReader()
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)

    record_defaults = [[1], ['unknown'], [1]]
    col1, col2, col3 = tf.decode_csv(value, record_defaults=record_defaults)
    features = tf.stack([col1, col3])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            step = 0
            while not coord.should_stop():
                print("current threadL{}".format(threading.current_thread()))
                step += 1
                example, label = sess.run([features, col3])
                print("line:", step, example, label)
        except tf.errors.OutOfRangeError:
            print(' training for {} epochs, {} steps'.format(NUM_EPOCHS,step))
        finally:
            coord.request_stop()
            coord.join(threads)

def get_data2():
    filename_queue = tf.train.string_input_producer(tables, num_epochs=NUM_EPOCHS)

    # reader = tf.TableRecordReader()
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)

    record_defaults = [[1], ['unknown'], [1]]
    col1, col2, col3 = tf.decode_csv(value, record_defaults=record_defaults)
    features = tf.stack([col1, col3])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            step = 0
            while not coord.should_stop():
                print("current threadL{}".format(threading.current_thread()))
                step += 1
                example, label = sess.run([features, col2])
                print("line:", step, example, label)
        except tf.errors.OutOfRangeError:
            print(' training for {} epochs, {} steps'.format(NUM_EPOCHS,step))
        finally:
            coord.request_stop()
            coord.join(threads)

get_data_1_thread=threading.Thread(target=get_data,name='thread [get_data_1]')
get_data_2_thread=threading.Thread(target=get_data2,name='thread [get_data_2]')
get_data_1_thread.start()
get_data_2_thread.start()
get_data_1_thread.join()
get_data_2_thread.join()
print('aaaaaaa')