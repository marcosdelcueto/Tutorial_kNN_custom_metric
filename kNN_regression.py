#!/usr/bin/env python3
# Marcos del Cueto
# Import necessary libraries
import math
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor, DistanceMetric
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

### Function just used to test custom metrics ###
def mimic_minkowski(X1,X2):
    distance=0.0                            # Initialize distance
    for i in range(len(X1)):                # For each element in X1 (in this case it is just 1D)
        distance=distance+(X1[i]-X2[i])**2  # Make sum of squared differences
    distance=math.sqrt(distance)            # Calculate final distance as sqrt of previous sum
    return distance
### Function just used to test custom metrics ###
def custom_metric(X1,X2):
    diff = X1[0]-X2[0]                      # Calculate Day difference between X1 and X2
    diff = 360/365*diff                     # Transforms Day difference to angle difference
    diff = diff * math.pi/180               # Transform degree to radians
    distance = math.sqrt(1-math.cos(diff))  # Calculate distance in polar coordinates
    return distance
###################### MAIN CODE ######################
# Read data
db_file='dataset.csv'                       # Name of csv file with dataset
df=pd.read_csv(db_file,index_col=0)         # Read dataset into a dataframe
X = df['Day'].values                        # Assign 'Day' descriptor
y = df['T (deg C)'].values                  # Assign 'T (deg C)' target property
# kNN regression
Neighbors=[1,2,3,4,5,6,7,8,9,10]            # Specify number of neighbors k used in grid search
# Grid search loop
for k in Neighbors:
    # Initialize lists
    y_predicted=[]
    y_real=[]
    # Specify options for kNN regression
    kNNreg = KNeighborsRegressor(n_neighbors=k, weights='distance', metric=custom_metric)
    # Leave-one-out loop
    for train_index, test_index in LeaveOneOut().split(X):
        # Assign train/test values
        X_train,X_test=X[train_index],X[test_index]
        y_train,y_test=y[train_index],y[test_index]
        X_train = X_train.reshape(-1, 1)
        X_test = X_test.reshape(-1, 1)
        # Predict data
        y_pred = kNNreg.fit(X_train, y_train.ravel()).predict(X_test)
        # Append data of each leave-one-out iteration
        y_predicted.append(y_pred.tolist())
        y_real.append(y_test.tolist())
    # Flatten lists with real and predicted values
    y_real = [item for dummy in y_real for item in dummy ]
    y_predicted = [item for dummy in y_predicted for item in dummy ]
    # Calculate r and rmse metrics
    r, _ = pearsonr(y_real, y_predicted)
    rmse  = math.sqrt(mean_squared_error(y_real, y_predicted))
    # Print results for each k value in the grid search
    print('Neighbors: %i. r: %.3f. rmse: %.3f' %(k, r, rmse))
