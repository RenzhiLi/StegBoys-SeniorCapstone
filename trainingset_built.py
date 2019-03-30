import Preprocessing
import numpy as np 
import pandas as pd
import sklearn
import sys

def built(good,bad,output,resize):
    open(output, 'w').close()
    Preprocessing.built_training_set(good,output,resize,0)
    Preprocessing.built_training_set(bad,output,resize,1)

def cmd_exec():
    built(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])

if __name__ == "__main__":
    built('./Inputpics/train/Good','./Inputpics/train/Bad','sample_trainingset.csv',(64,64))
