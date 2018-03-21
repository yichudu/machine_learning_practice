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
    train_op, loss_op = model_fn(features_placeholder, labels_placeholder, mode=tf.saved_model.tag_constants.TRAINING)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # 训练模型
        STEPS = 1000
        loss_arr = []
        for i in range(STEPS):
            _,loss_evl = sess.run([train_op, loss_op], feed_dict={features_placeholder: features, labels_placeholder: labels})
            if i%100 ==0:
                print(loss_evl)
        builder.add_meta_graph_and_variables(sess,tags=tf.saved_model.tag_constants.SERVING)

    builder.save()