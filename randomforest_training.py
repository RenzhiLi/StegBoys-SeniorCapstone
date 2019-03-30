import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
train_df=pd.read_csv('sample_trainingset.csv')
train=np.array(train_df)
train_x=train[:,1:-1]
train_x=preprocessing.scale(train_x)
train_y=train[:,-1]
train_y=np.int32(train_y)
x_train,x_test,y_train,y_test=train_test_split(train_x,train_y,test_size=0.1,random_state=0)


rfc=RandomForestClassifier(n_estimators=250,n_jobs=-1,max_features=10,random_state=0)
rfc.fit(x_train,y_train)
y_predit=rfc.predict(x_test)
print(metrics.accuracy_score(y_test,y_predit))

from joblib import dump, load
dump(rfc,'./modelrfc.joblib')