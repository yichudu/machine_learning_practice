import tensorflow as tf

LOGDIR='d:/tf-logdir/'
import os
import time


out_dir = LOGDIR+'demo'

x = tf.placeholder(tf.float32, name="x")

init_op = tf.global_variables_initializer()

summary_op = tf.summary.scalar("steps", x)

with tf.Session() as sess:
  sess.run(init_op)
  summary_writer = tf.summary.FileWriter(out_dir, sess.graph)
  for step in range(10):
    summary = sess.run(summary_op, feed_dict={x: step})
    summary_writer.add_summary(summary, step)