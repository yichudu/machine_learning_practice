import tensorflow as tf

from sklearn import datasets
from sklearn import model_selection
import os
from os.path import isdir, isfile, join

import numpy as np
from tensorflow_practice.BP.iris_predict_api import *

if __name__ == '__main__':
    export_dir = 'd:/tmp/model_save_restore/'

    # delete old files
    if isdir(export_dir):
        import shutil

        shutil.rmtree(export_dir)
        # os.rmdir(export_dir)
        delete_flag = True
        print('delete OK!')

    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    train_op, loss_tensor, predict_tensor = model_fn(features_placeholder, labels_placeholder, mode=None)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # 训练模型
        STEPS = 1000
        loss_arr = []
        for i in range(STEPS):
            _, loss_evl = sess.run([train_op, loss_tensor],
                                   feed_dict={features_placeholder: features, labels_placeholder: labels})
            if i % 100 == 0:
                print(loss_evl)

        features_placeholder_info = tf.saved_model.utils.build_tensor_info(features_placeholder)
        predict_info = tf.saved_model.utils.build_tensor_info(predict_tensor)

        signature_def = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'features': features_placeholder_info}, outputs={'predict': predict_info},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        builder.add_meta_graph_and_variables(sess, tags=[tf.saved_model.tag_constants.SERVING], signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def})

    builder.save()
