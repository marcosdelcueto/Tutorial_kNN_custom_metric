#!/usr/bin/env python3
# Marcos del Cueto
import math
from math import sqrt
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor, DistanceMetric
from sklearn.model_selection import StratifiedKFold, train_test_split, learning_curve, LeaveOneOut, KFold
from sklearn.metrics import mean_squared_error,make_scorer
from scipy.stats import pearsonr, spearmanr


### Function just used to test custom metrics ###
def mimic_minkowski(X1,X2):
    distance=0.0
    #print('new call minkowski')
    #print('X1:')
    #print(X1)
    #print('X2:')
    #print(X2)
    for i in range(len(X1)):
        distance=distance+(X1[i]-X2[i])**2
    distance=distance**(1.0/2.0)
    #print('distance:', distance)
    #print('##################')
    return distance

### Function just used to test custom metrics ###
def custom_metric(X1,X2):
    diff=0.0
    for i in range(len(X1)):
        diff=diff+(X1[i]-X2[i])**2
    diff=diff**(1.0/2.0)
    #print('diff:', diff)
    #print('##################')
    diff = 360/365*diff
    diff = diff * math.pi/180
    distance = sqrt(150e6*(1-math.cos(diff)))
    return distance

db_file='canada_average.csv'
Neighbors=[1,2,3,4,5,6,7,8,9,10]
kfold=50
df=pd.read_csv(db_file,index_col=0)

print (df)

X = df['Day'].values
y = df['T (deg C)'].values
#print('X:')
#print(X)
#print('y:')
#print(y)

X.reshape(-1, 1)
y.reshape(-1, 1)





for k in Neighbors:
    print('NEW NEIGHBOR',k)
    #ML_algorithm = KNeighborsRegressor(n_neighbors=k, weights='distance')
    #ML_algorithm = KNeighborsRegressor(n_neighbors=k, weights='distance', metric=mimic_minkowski)
    ML_algorithm = KNeighborsRegressor(n_neighbors=k, weights='distance', metric=custom_metric)
    y_predicted=[]
    y_real=[]
    kf = KFold(n_splits=kfold,shuffle=True)
    validation=kf.split(X)
    loo = LeaveOneOut()
    validation=loo.split(X)
    for train_index, test_index in validation:
        #print('NEW FOLD')
        X_train,X_test=X[train_index],X[test_index]
        y_train,y_test=y[train_index],y[test_index]
        X_train = X_train.reshape(-1, 1)
        X_test = X_test.reshape(-1, 1)
        y_pred = ML_algorithm.fit(X_train, y_train.ravel()).predict(X_test)
        y_predicted.append(y_pred.tolist())
        y_real.append(y_test.tolist())
    
    y_real = [item for dummy in y_real for item in dummy ]
    y_predicted = [item for dummy in y_predicted for item in dummy ]
    #print('y_predicted:', y_predicted)
    #print('y_real:', y_real)
    
    r, _ = pearsonr(y_real, y_predicted)
    rms  = sqrt(mean_squared_error(y_real, y_predicted))
    print('Neighbors:',k,'r:',r,'rmse:',rms)
