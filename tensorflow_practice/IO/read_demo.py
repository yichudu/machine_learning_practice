import tensorflow as tf
import time
import threading



tables = ['d:/in/a.csv']
NUM_EPOCHS=1
filename_queue = tf.train.string_input_producer(tables, num_epochs=NUM_EPOCHS)

# reader = tf.TableRecordReader()
reader = tf.TextLineReader(skip_header_lines=1)
key, value = reader.read(filename_queue)

content_batch_queue = tf.train.batch([value],
                                     batch_size=3,
                                     num_threads=2,
                                     capacity=5 * 2,
                                     allow_smaller_final_batch=True
                                     )


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
        step = 0
        while not coord.should_stop():
            step += 1
            [example_arr_tensor] = sess.run([content_batch_queue])
            for sample in example_arr_tensor:
                print("step:{}, sample:{}, type:{}".format( step, sample.decode('utf-8'),type(sample)))
    except tf.errors.OutOfRangeError:
        print(' training for {} epochs, {} steps'.format(NUM_EPOCHS,step))
    finally:
        coord.request_stop()
        coord.join(threads)

"""
step:1, sample:小明,13,88, type:<class 'bytes'>
step:1, sample:小红,12,99, type:<class 'bytes'>
step:1, sample:小刚,14,89, type:<class 'bytes'>
step:2, sample:A,12,90.33333333, type:<class 'bytes'>
step:2, sample:B,23,89.83333333, type:<class 'bytes'>
step:2, sample:C,11,89.33333333, type:<class 'bytes'>
step:3, sample:D,22,88.83333333, type:<class 'bytes'>
step:3, sample:E,13,88.33333333, type:<class 'bytes'>
step:3, sample:F,21,87.83333333, type:<class 'bytes'>
step:4, sample:G,24,87.33333333, type:<class 'bytes'>
step:4, sample:H,25,86.83333333, type:<class 'bytes'>
step:4, sample:I,25,86.33333333, type:<class 'bytes'>
step:5, sample:J,26,85.83333333, type:<class 'bytes'>
step:5, sample:K,26,85.33333333, type:<class 'bytes'>
 training for 1 epochs, 6 steps
"""