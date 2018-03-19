
import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
import csv


df_train = pd.read_csv('C:\Users\Charlotte\PycharmProjects\Titanic_project\data\\train.csv')
df_test = pd.read_csv('C:\Users\Charlotte\PycharmProjects\Titanic_project\data\\test.csv')

print df_train.shape
print df_test.shape
# print df_train.isnull().sum() #null feature sum: Age:177, Cabin:687, Embarked:2
# print df_test.isnull().sum()

#preprocessing----------------------------------------------------------------------------------------------------------
def preprocessing(df):

    # print df.isnull().sum() # null feature sum: Age:177, Cabin:687, Embarked:2

    df['Age'].fillna(-1, inplace=True)
    df['Fare'].fillna(-1, inplace=True)
    df['Cabin'].fillna('U', inplace=True) #people without a cabin recorded were probably similar
    df['Embarked'].fillna('S', inplace=True) #only 2 missing values, but dont want to lose data





    f = lambda x: x.split(',')[1].split('.')[0]
    df['Name'] = df['Name'].apply(f)

    titles = df['Name'].unique()


    #check if mean age changes with title

    means_age = dict()
    for title in titles:
        mean_age = df.Age[(df['Age'] != -1) & (df['Name'] == title)].mean()
        means_age[title] = mean_age

    import math
    if math.isnan(means_age[' Ms']):
        means_age[' Ms'] = means_age[' Mr']


    for index, row in df.iterrows():
        if row['Age'] == -1:
            # print index
            df.loc[index, 'Age'] = means_age[row['Name']]

    means_fare = dict()
    for pclass in df['Pclass'].unique():
        mean_fare = df.Fare[(df['Fare'] != -1) & (df['Pclass'] == pclass)].mean()
        means_fare[pclass] = mean_fare

    for index, row in df.iterrows():
        if row['Fare'] == -1:
            df.loc[index, 'Fare'] = means_fare[row['Pclass']]

    df['Cabin'] = df['Cabin'].apply(lambda x: x[0])

    df = pd.get_dummies(df,columns= ['Sex'])

    # null_columns = df.columns[df.isnull().any()]
    # df[null_columns].isnull().sum()
    # print 'after:' + str(df[df['Age'].isnull()][null_columns])
    # print 'new values:'
    # print df.loc[88,'Age']

    print len(df.columns)

    return df

df_train = preprocessing(df_train)
df_test = preprocessing(df_test)

X_train = df_train.drop(['PassengerId', 'Ticket', 'Name', 'Cabin','Embarked', 'Survived'], axis=1)
X_test = df_test.drop(['PassengerId', 'Ticket', 'Cabin','Embarked', 'Name'], axis=1)
y = df_train['Survived']

#make feature number the same for both training and test set

train_objs_num = len(X_train)
X_train_test = pd.concat(objs = [X_train, X_test], axis = 0)
X_train_test = pd.get_dummies(X_train_test)
X_train = copy.copy(X_train_test[:train_objs_num])
X_test = copy.copy(X_train_test[train_objs_num:])
X_test.fillna(0, inplace = True)
X_train.fillna(0, inplace = True)


X_train_std = StandardScaler().fit_transform(X_train)
X_test_std = StandardScaler().fit_transform(X_test)


#number of features for test and training set are different, can't use svm wout some sort of dimensionality reduction
svm = SVC(kernel = 'rbf', C=1000, gamma = 0.005)
svm.fit(X_train_std,y)
# predict_svm = svm.predict(X_test_std)

# forest = RandomForestClassifier(n_estimators = 20, criterion = 'entropy',max_features= 10, max_depth = None, min_samples_split= 3, min_samples_leaf= 3, bootstrap= True)
# forest.fit(X_train,y)
# predict_forest = forest.predict(X_test)


#export test file results-----------------------------------------------------------------------------------------------

csvfile = "C:\Users\Charlotte\PycharmProjects\Titanic_project\submissions\\18.3.18\\test_6.csv"

#Assuming res is a flat list
# with open(csvfile, "w") as output:
#     writer = csv.writer(output, lineterminator='\n')
#     for val in predict_svm:
#         writer.writerow([val])

#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------

#testing----------------------------------------------------------------------------------------------------------------
#quantify how well the grid search parameters characterize the test set (what it hasn't seen before)
def testing(clf,X):

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size = 0.25,
                                                       random_state=42)


    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print str(clf) + 'bulk score %.3f' %score

    scores = cross_val_score(estimator=clf,
                             X = X_test,
                             y=y_test,
                             cv=5,
                             n_jobs=1)
    print str(clf) + 'CV accuracy scores: %s' %scores
    print str(clf) + 'CV accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores))

svm = SVC(kernel = 'rbf', C=1000, gamma = 0.005)
forest = RandomForestClassifier(n_estimators = 20, criterion = 'entropy',max_features= 10, max_depth = None, min_samples_split= 3, min_samples_leaf= 3, bootstrap= True)
testing(svm,X_train_std) #standardize before svm
# testing(forest,X_train) #don't standardize before random forest
#-----------------------------------------------------------------------------------------------------------------------


#gridsearch-------------------------------------------------------------------------------------------------------------
def gridsearch(clf,X,param_grid_svm ):

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.3,
                                                        random_state=42)


    gs = GridSearchCV(clf, cv = 5, param_grid= param_grid_svm)

    gs.fit(X_train, y_train)
    print str(clf) + 'gridsearch %f' %gs.best_score_
    print gs.best_params_


#svm search parameters
param_range_gamma = [0.0001,0.0005,0.001,0.005,0.01,0.1]
param_range_C = [1,1e2,1e3,1e4,1e5,1e6]
param_grid_svm = [ {'C': param_range_C,
                       'gamma': param_range_gamma,
                       'kernel': ['rbf']}]
svm = SVC(kernel = 'rbf')


#random forest parameters
param_grid_forest = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
forest = RandomForestClassifier(n_estimators = 20)

#Run gridsearch for different classifiers (randomized search vs gridsearch?)
# gridsearch(svm, X_train_std, param_grid_svm )
#best svm params for 'kernel': 'rbf' {'C': 1.0, 'gamma': 0.1}
#best svm params for 'kernel': 'rbf' {'kernel': 'rbf', 'C': 100.0, 'gamma': 0.01} ,after removing name feature
#best svm params for 'kernel': 'rbf' {'kernel': 'rbf', 'C': 1, 'gamma': 0.1}, after removing name and cabin features
#best svm params for 'kernel': 'rbf' {'kernel': 'rbf', 'C': 10000.0, 'gamma': 0.0005}, after removing name and cabin features and making pclass categorical
#best svm params for 'kernel': 'rbf' {'kernel': 'rbf', 'C': 1000.0, 'gamma': 0.005}, after removing name and making pclass categorical
#best svm params for 'kernel': 'rbf'{'kernel': 'rbf', 'C': 1, 'gamma': 0.1}, after removing name, cabin and embarked

# gridsearch(forest, X_train, param_grid_forest )
#best randomforest params  {'bootstrap': False, 'min_samples_leaf': 3, 'min_samples_split': 10, 'criterion': 'entropy', 'max_features': 3, 'max_depth': None}
#best randomforest params {'bootstrap': True, 'min_samples_leaf': 3, 'min_samples_split': 3, 'criterion': 'entropy', 'max_features': 3, 'max_depth': None}, after removing name feature
#best randomforest params {'bootstrap': False, 'min_samples_leaf': 10, 'min_samples_split': 10, 'criterion': 'entropy', 'max_features': 1, 'max_depth': None}, after removing name and cabin features
#best randomforest params{'bootstrap': True, 'min_samples_leaf': 3, 'min_samples_split': 2, 'criterion': 'entropy', 'max_features': 1, 'max_depth': None}, after removing name and cabin features and making pclass categorical
#best randomforest params{'bootstrap': True, 'min_samples_leaf': 3, 'min_samples_split': 3, 'criterion': 'entropy', 'max_features': 10, 'max_depth': None}, after removing name and making pclass categorical
#-----------------------------------------------------------------------------------------------------------------------

def RandomForest(X_train,y_train):
    forest = RandomForestClassifier(n_estimators=1000,
                                    random_state = 0,
                                    n_jobs=-1)
    forest.fit(X_train,y_train)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    feat_labels = X_train.columns

    for f in range(X_train.shape[1]):
             print '%2d %-*s %f' % (f + 1,30,
                                    feat_labels[indices[f]],
                                    importances[indices[f]])


# RandomForest(X_train,y)