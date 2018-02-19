# -*- coding: utf-8 -*-
"""
Title: iris.py
Date: February 17th, 2018
Author: Matt Runyon
Description: This script explores the iris flower data set. This is a test
             case typical for many statistical classification techniques in
             machine learning. The iris data is apparently the 'hello world'
             of machine learning.

             The procedure followed here was taken from the following URL:
                 https://machinelearningmastery.com/
                     machine-learning-in-python-step-by-step/
"""

"""
#============================================================================#
#                       IMPORTS                                              #
#============================================================================#
"""

import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

"""
#============================================================================#
#                       CONSTANTS                                            #
#============================================================================#
"""
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
num_rows_to_peak = 20
test_size = 0.20
seed = 7
k = 10
scoring = 'accuracy'
"""
#============================================================================#
#                       ANALYZE DATA                                         #
#============================================================================#
"""

# retrieve data online
dataset = pandas.read_csv(url, names=names)

# determine the number of rows and columns (shape) in the data
num_rows = dataset.shape[0]
num_cols = dataset.shape[1]
print("There are %d rows and %d columns in the dataset" % (num_rows, num_cols))

# take a look with 'head'
print("Here are the first %d entries:\n" % num_rows_to_peak)
print(dataset.head(num_rows_to_peak))

# Get statistical summary
print(dataset.describe())

# create plots for visualization
plt.figure()
dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)

plt.figure()
dataset.hist()

plt.figure()
scatter_matrix(dataset)

# Get validation dataset and train data set
array = dataset.values
X = array[:,0:4]
Y = array[:,4]

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(
        X, Y, test_size=test_size, random_state=seed)

"""
#============================================================================#
#                       BUILD A MODEL                                        #
#============================================================================#
"""
# These are six different models we will try out. Each model will attempt
# to create some mathematical function that maps input data (the iris data)
# to output data (the iris species). The mathematics will be different for
# each model.
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

"""
Evaluate each model through k-fold cross validation. Here, the iris data is 
partitioned into k subsets. k-1 subsets are used as training data for each
model to construct a mapping and the last subset is used as the 'test set' 
for each model so we can see how well it can guess what an iris species is
based on measurements of a particular iris. The metric to determine 'how well'
is given by <scoring>.
"""
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=k, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train,
                                                 cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
