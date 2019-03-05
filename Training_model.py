import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load


data = pd.read_csv('/Users/mohammedawan/PycharmProjects/StegModel/sample_trainingset.csv').values
dim = len(data[0])
Y = data[:,dim-1:].ravel().astype('int')
X = data[:,:dim-1]
train_x, test_x, train_y, test_y = train_test_split(X, Y)
filenames_test = test_x[:,:1]
data_test = test_x[:,1:]
data_train = train_x[:,1:]

clf = RandomForestClassifier(max_depth=10, criterion='entropy')
clf.fit(data_train,train_y)

dump(clf, '/Users/mohammedawan/PycharmProjects/StegModel/stemodel.joblib')
clf1 = load('/Users/mohammedawan/PycharmProjects/StegModel/stemodel.joblib')
prediction = clf1.predict(data_test)

print(accuracy_score(test_y, prediction))