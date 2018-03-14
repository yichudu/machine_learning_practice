"""
通过 tf.train.Saver 来持久化变量数据
"""
import tensorflow as tf
import os
from os.path import isdir, isfile, join

tf.reset_default_graph()

# Create some variables.
v1 = tf.get_variable("v1", shape=[3], initializer=tf.zeros_initializer)
v2 = tf.get_variable("v2", shape=[5], initializer=tf.zeros_initializer)

inc_v1_op = v1.assign(v1 + 1)
dec_v2_op = v2.assign(v2 - 1)

# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

path = 'd:/tmp/'


def run_and_save():
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Later, launch the model, initialize the variables, do some work, and save the
    # variables to disk.
    with tf.Session() as sess:
        sess.run(init_op)
        # Do some work with the model.

        print(sess.run(inc_v1_op)) # type(sess.run(inc_v1_op)) is  numpy.ndarray
        print(sess.run(dec_v2_op))

        # delete old files
        delete_flag=False
        for file in os.listdir(path):
            if isfile(join(path, file)) and (file.find('model') >= 0 or file.find('checkpoint') >= 0):
                os.remove(join(path, file))
                delete_flag=True
        if delete_flag:
            print('delete OK!')

        # Save the variables to disk.
        save_path = saver.save(sess, join(path, "model.ckpt"))
        print("Model saved in file: %s" % save_path)


def restore():
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        print('before restore:')
        print("v1 : %s" % v1.eval())
        print("v2 : %s \n" % v2.eval())
        saver.restore(sess, join(path, "model.ckpt"))
        print("after restore ")
        # Check the values of the variables
        print("v1 : %s" % v1.eval())
        print("v2 : %s" % v2.eval())


run_and_save()
#tf.reset_default_graph()
restore()
"""
[ 1.  1.  1.]
[-1. -1. -1. -1. -1.]
delete OK!
Model saved in file: d:/tmp/model.ckpt
before restore:
v1 : [ 0.  0.  0.]
v2 : [ 0.  0.  0.  0.  0.] 

after restore 
v1 : [ 1.  1.  1.]
v2 : [-1. -1. -1. -1. -1.]

"""