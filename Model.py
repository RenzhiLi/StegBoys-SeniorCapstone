import sys
import numpy as np
import pandas as pd
from joblib import dump, load


name = sys.argv[1]
data = pd.read_csv(name, header=None)
filenames = data.values[:,:1]
data = data.values[:,1:]

clf = load('stemodel.joblib')
prediction = clf.predict(data)

good = 0
total = len(prediction)
for file, pred in zip(filenames, prediction):
    if(pred == 0):
        good += 1
        print(file)
bad = total - good

print('Total Num of Pics: ' + str(total) + '\n Precentage of Good: ' + str(good/total) + '\n Precentage of Bad: ' + str(bad/total))
