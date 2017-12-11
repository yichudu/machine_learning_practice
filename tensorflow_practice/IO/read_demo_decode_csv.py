# coding:utf-8
import tensorflow as tf
import time
import threading

tables = ['d:/in/a.csv']
NUM_EPOCHS = 2
filename_queue = tf.train.string_input_producer(tables, num_epochs=NUM_EPOCHS)

# reader = tf.TableRecordReader()
reader = tf.TextLineReader(skip_header_lines=1)
key, value = reader.read(filename_queue)

record_defaults = [['name'], [1], [1.]]
csv_value = tf.decode_csv(value, record_defaults=record_defaults)
# 不能写tf.train.batch([csv_value]), 类型不一样的要分开写
batch_csv_value_op = tf.train.batch([csv_value[0],csv_value[1],csv_value[2]], batch_size=3, num_threads=2, capacity=5 * 2,allow_smaller_final_batch=True )

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
        step = 0
        while not coord.should_stop():
            step += 1
            [name_arr,age_arr,score_arr] = sess.run(batch_csv_value_op)
            for i in range(len(name_arr)):
                name = name_arr[i].decode('utf-8')
                age = age_arr[i]
                score = score_arr[i]
                print("step:{}, name:{}, age:{}, score:{}".format( step, name, age, score))
    except tf.errors.OutOfRangeError:
        print(' training for {} epochs, {} steps'.format(NUM_EPOCHS, step))
    finally:
        coord.request_stop()
        coord.join(threads)

"""
step:1, name:小明, age:13, score:88.0
step:1, name:小刚, age:14, score:89.0
step:1, name:小红, age:12, score:99.0
step:2, name:A, age:12, score:90.33333587646484
step:2, name:B, age:23, score:89.83333587646484
step:2, name:C, age:11, score:89.33333587646484
step:3, name:D, age:22, score:88.83333587646484
step:3, name:E, age:13, score:88.33333587646484
step:3, name:F, age:21, score:87.83333587646484
step:4, name:G, age:24, score:87.33333587646484
step:4, name:H, age:25, score:86.83333587646484
step:4, name:I, age:25, score:86.33333587646484
step:5, name:J, age:26, score:85.83333587646484
step:5, name:K, age:26, score:85.33333587646484
step:5, name:小明, age:13, score:88.0
step:6, name:小红, age:12, score:99.0
step:6, name:小刚, age:14, score:89.0
step:6, name:A, age:12, score:90.33333587646484
step:7, name:C, age:11, score:89.33333587646484
step:7, name:B, age:23, score:89.83333587646484
step:7, name:D, age:22, score:88.83333587646484
step:8, name:E, age:13, score:88.33333587646484
step:8, name:F, age:21, score:87.83333587646484
step:8, name:G, age:24, score:87.33333587646484
step:9, name:H, age:25, score:86.83333587646484
step:9, name:I, age:25, score:86.33333587646484
step:9, name:J, age:26, score:85.83333587646484
step:10, name:K, age:26, score:85.33333587646484
training for 2 epochs, 11 steps

"""