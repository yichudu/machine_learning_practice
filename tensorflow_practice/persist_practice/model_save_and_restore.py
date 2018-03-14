import tensorflow as tf

from sklearn import datasets
from sklearn import model_selection
import os
from os.path import isdir, isfile, join

import numpy as np
from tensorflow_practice.BP.iris_predict_api import *

if __name__ =='__main__':
    export_dir = 'd:/tmp/model_save_restore/'

    # delete old files
    if isdir(export_dir) :
        os.rmdir(export_dir)
        delete_flag = True
        print('delete OK!')

    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

    with tf.Session(graph=tf.Graph()) as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # 训练模型
        STEPS = 1000
        loss_arr = []
        for i in range(STEPS):
            loss_evl, _ = sess.run([loss, train], feed_dict={x: X_train, y: y_train})
            if i%100 ==0:
                print(loss_evl)
        builder.add_meta_graph_and_variables(sess)

    builder.save()