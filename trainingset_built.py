import Preprocessing
import numpy as np 
import pandas as pd
import sklearn

def built(good,bad,output,resize):
    open(output, 'w').close()
    Preprocessing.built_training_set(good,output,(64,64),0)
    Preprocessing.built_training_set(bad,output,(64,64),1)

if __name__ == "__main__":
    built('./Inputpics/Good','./Inputpics/Bad','sample_trainingset.csv',(64,64))
