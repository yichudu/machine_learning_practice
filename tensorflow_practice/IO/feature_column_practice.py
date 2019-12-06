import pandas as pd
import tensorflow as tf
from tensorflow.contrib import layers


def get_dataset():
    data = [['Tom', 'M', 95], ['Jerry', 'M', 96], ['Tonny', 'M', 97], ['Lisa', 'F', 98]]
    dataframe = pd.DataFrame(data=data, columns=['name', 'gender', 'score'])
    labels = dataframe.pop('score')
    features = dict()
    for col_name in dataframe.columns:
        features[col_name] = dataframe[col_name].tolist()

    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.batch(2)
    return dataset


def get_feature_column():
    feature_name = 'gender'
    sparse_id_column = layers.sparse_column_with_hash_bucket(
        column_name=feature_name,
        hash_bucket_size=100
    )
    feature_column = layers.embedding_column(
        sparse_id_column,
        dimension=10
    )
    return feature_column

features, _labels = get_dataset().make_one_shot_iterator().get_next()
feature_column = get_feature_column()
result = layers.input_from_feature_columns(features, [feature_column])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2):
        result_eval = sess.run(fetches=[result])
        print(result_eval)

"""
[array([[-0.03351991,  0.13861407,  0.15951617, -0.10525074, -0.02098984,
         0.11429874,  0.07259871, -0.05891977,  0.13090345, -0.04620567],
       [-0.03351991,  0.13861407,  0.15951617, -0.10525074, -0.02098984,
         0.11429874,  0.07259871, -0.05891977,  0.13090345, -0.04620567]],
      dtype=float32)]
[array([[-0.03351991,  0.13861407,  0.15951617, -0.10525074, -0.02098984,
         0.11429874,  0.07259871, -0.05891977,  0.13090345, -0.04620567],
       [-0.00928837, -0.06804372,  0.10571972, -0.18538876, -0.11762749,
        -0.03500816, -0.01680218, -0.08379114,  0.0384082 , -0.07948529]],
      dtype=float32)]
"""
