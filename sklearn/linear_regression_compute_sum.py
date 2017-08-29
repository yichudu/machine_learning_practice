import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from util import yc_io_util as yc


def generate_data(rows):
    data = np.random.randint(0, 100, size=(rows, 2))
    df = DataFrame(data=data, columns=['num1', 'num2']);
    df['sum'] = df['num1'] + df['num2']
    yc.create_file('d:/out/ml/train.csv')
    df.to_csv('d:/out/ml/train.csv', index=False)


def step_train(X, y):
    reg_model = LinearRegression()
    reg_model.fit(X, y)
    reg_model_score = reg_model.score(X, y)
    print('reg_model_score:\t', reg_model_score)
    print("reg_model's coefficient:\t", reg_model.coef_)
    return reg_model

def step_predict(df, ml_model):
    pred=ml_model.predict(df)
    pred_df=DataFrame({'num1':df.loc[:,'num1'],\
                       'num2': df.loc[:, 'num2'],\
                       'predict_sum':pred.T})
    pred_df.to_csv('d:/out/ml/predict.csv', index=False)

generate_data(100)
train = pd.read_csv("d:/out/ml/train.csv")
training, testing = train_test_split(train, test_size=0.2, random_state=0)
training_X = training.loc[:, ['num1', 'num2']]
training_y = training['sum']
testing_X = testing.loc[:, ['num1', 'num2']]
testing_y = testing['sum']

print("Total sample size = %i; training sample size = %i, testing sample size = %i" \
      % (train.shape[0], training.shape[0], testing.shape[0]))
X = training.loc[:, ['num1', 'num2']]
y = training['sum']


classifier_LR=step_train(training_X, training_y)
step_predict(testing_X,classifier_LR)

"""
Total sample size = 100; training sample size = 80, testing sample size = 20
reg_model's coefficient:     array([ 1.,  1.])
reg_model_score:	 1.0

"""