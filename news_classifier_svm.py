# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 19:36:37 2019

使用合适的方法，对文本进行分类，看是否是新华社的文章

@author: us
"""
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

#获取数据
fileHandle = open('D:/开课吧/NLP11/sen2vec_news.file', 'rb')
corp = pickle.load(fileHandle)
fileHandle.close()

# svm test 借鉴SVM包的使用方法
'''
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])

clf = SVC()
clf.fit(X, y) 
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
print(clf.predict([[-0.8, -1]]))
'''

#整理数据
corp = corp[['result','sen2vec']]
corp['sen2vec'] = corp['sen2vec'].apply(lambda x:list(x)[0])
corp_x = corp['sen2vec'].tolist() 
corp_x = [list(x) for x in corp_x]
corp_x = np.array(corp_x)
corp_y = np.array(corp['result'])

#分训练集、测试集
X_train, X_test, Y_train, Y_test = train_test_split(corp_x, corp_y, test_size=0.25)

#模型训练
clf = SVC()
print('is training...')
clf.fit(X_train, Y_train) 
print('finish training!')
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

#模型评价
Y_prediction = clf.predict(X_train)
traing_accuracy = 1-sum(abs(Y_train-Y_prediction))/len(Y_train)
Y_prediction_t = clf.predict(X_test)    
test_accuracy = 1-sum(abs(Y_test-Y_prediction_t))/len(Y_test)
   
d = {
     "training_accuracy": traing_accuracy,
     "test_accuracy":test_accuracy,
     }
print(d)
'''
{'training_accuracy': 0.9032149366662072, 'test_accuracy': 0.9047925377935028}
'''