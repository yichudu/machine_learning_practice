import tensorflow as tf
import pandas as pd
def get_dataset():
    data = [['Tom', 'M', 95], ['Jerry', 'M', 96], ['Tonny', 'M', 97], ['Lisa', 'F', 98]]
    dataframe = pd.DataFrame(data=data, columns=['name', 'gender', 'score'])
    labels = dataframe.pop('score')
    features = dict()
    # call `pd.Series#tolist()` apparently
    for col_name in dataframe.columns:
        features[col_name] = dataframe[col_name].tolist()
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset=dataset.batch(2)
    return dataset

dataset=get_dataset()
features,labels = dataset.make_one_shot_iterator().get_next()

with tf.Session() as sess:
    print(sess.run([features,labels]))
    print(sess.run([features,labels]))
"""
[({'name': array([b'Tom', b'Jerry'], dtype=object), 'gender': array([b'M', b'M'], dtype=object)}, array([95, 96], dtype=int64))]
[({'name': array([b'Tonny', b'Lisa'], dtype=object), 'gender': array([b'M', b'F'], dtype=object)}, array([97, 98], dtype=int64))]
"""
