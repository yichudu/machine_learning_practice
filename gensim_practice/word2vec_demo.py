from gensim.models import Word2Vec
# import modules & set up logging
import gensim, logging

#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

documents = ["Human machine interface for lab abc computer applications",
              "A survey of user opinion of computer system response time",
              "The EPS user interface management system",
              "System and human system engineering testing of EPS",
              "Relation of user perceived response time to error measurement",
              "The generation of random binary unordered trees",
              "The intersection graph of paths in trees",
              "Graph minors IV Widths of trees and well quasi ordering",
              "Graph minors A survey"]

sentences=[]
for sentence in documents:
    sentence_arr=[]
    for word in sentence.split(' '):
        sentence_arr.append(word)
    sentences.append(sentence_arr)

print(sentences)
# train word2vec on the 9 sentences
model = Word2Vec(sentences, min_count=0)

#gensim.models.word2vec.Text8Corpus

print(model['computer'])
"""
# 每次运行结果不一样
[  2.73460872e-03  -4.65013599e-03  -3.80138517e-04   2.23577931e-03
   3.53459432e-03  -3.34876822e-04   1.06677762e-03  -1.45550934e-03
  -3.63708055e-03  -1.03985704e-03   5.88597555e-04  -3.29646142e-03
  -3.69138992e-03  -2.41321931e-03  -1.03089481e-03   2.81171687e-03
   1.55196059e-03   4.87440825e-03   1.50484429e-03   4.80763993e-05
  -1.87746075e-03   9.91601613e-04  -2.17233296e-03  -4.53947438e-03
  -1.77033740e-04  -2.63292203e-03   7.17394112e-04  -3.50880786e-03
   4.32372605e-03  -4.97234333e-03  -3.12788039e-03   4.44929721e-03
   7.83776486e-05  -4.06554714e-03   8.32864782e-04  -9.04903631e-04
  -6.80863450e-04  -5.64683229e-04   4.03504586e-03   1.44853489e-03
  -1.34819082e-03   9.88882268e-04   4.90686949e-03  -4.15528950e-04
  -3.84552637e-03   2.02916353e-03  -2.32719103e-04   4.46808105e-03
  -2.41433899e-03   3.47174006e-03  -6.23397529e-04  -1.73202623e-03
   1.80823647e-03   1.34497508e-03  -1.86103873e-03  -2.03344948e-03
   3.99009045e-03  -3.43435118e-03  -1.21433951e-03   3.11270007e-03
  -4.25252970e-03  -2.64853850e-04   7.29947060e-04  -3.04733706e-03
   4.27079201e-03   2.69243796e-03   1.19566231e-03  -4.84854681e-03
   2.37741711e-04  -4.44092648e-03  -2.63062282e-03  -1.60635286e-03
   4.23326530e-03   3.95722268e-03   3.40898172e-03  -3.91058763e-03
  -2.33103638e-03  -2.18217564e-03  -3.02792713e-03  -1.19338324e-03
   2.37075426e-03  -3.08191702e-05   3.91526101e-03   3.40931839e-03
  -4.86839330e-03  -1.28278451e-03   3.01979342e-03  -9.92975314e-04
  -3.96849774e-03   4.93339263e-04  -1.31043978e-03   1.55910791e-03
   1.13663205e-03   1.81729032e-03   2.57179327e-03   4.35575470e-03
  -4.58758790e-03  -2.61309231e-03   1.08286226e-03   2.10967869e-03]
"""