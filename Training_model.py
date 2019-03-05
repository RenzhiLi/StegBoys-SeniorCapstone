import sys
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load

#Reads in data into a 2D array, Includes Filenames, RGB Values, and the class labels
data = pd.read_csv('/Users/mohammedawan/PycharmProjects/StegModel/sample_trainingset.csv').values
#Gets number of dimensions
dim = len(data[0])

#Splits data into class labels, Y, and the X (Filenames, RGB values)
Y = data[:,dim-1:].ravel().astype('int')
X = data[:,:dim-1]

#Splitst into testing and training data
train_x, test_x, train_y, test_y = train_test_split(X, Y)

#Seperate Filenames from RGB Values
filenames_test = test_x[:,:1]
#Test Data(RGB)
data_test = test_x[:,1:]
#Training Data(RGB)
data_train = train_x[:,1:]

#Defines the model and parameteres
clf = SVC()
#Fit the model to the data
clf.fit(data_train,train_y)

#Creates file with saved model
dump(clf, 'stemodel.joblib')
#Loads model from file
clf1 = load('stemodel.joblib')
#Runs a prediction on testing data
prediction = clf1.predict(data_test)
#Gives the accuracy of prediction from model
print(accuracy_score(test_y, prediction))
