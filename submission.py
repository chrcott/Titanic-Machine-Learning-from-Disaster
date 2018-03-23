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

def RandomForest(X_train,y_train):
    forest = RandomForestClassifier(n_estimators=20,
                                    random_state = 42,
                                    n_jobs=-1)
    forest.fit(X_train,y_train)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    feat_labels = X_train.columns

    for f in range(X_train.shape[1]):
             print '%2d %-*s %f' % (f + 1,30,
                                    feat_labels[indices[f]],
                                    importances[indices[f]])
def feature_cat(feature, df):
    feature_count = df.groupby(feature).PassengerId.count()
    feature_sum = df.groupby(feature).Survived.sum()

    percent_survived = feature_sum/feature_count*100

    print percent_survived
    # percent_survived.plot(kind='bar')

    plt.show()
def testing(clf, X):
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.25,
                                                            random_state=42)

        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print str(clf) + 'bulk score %.3f' % score

        scores = cross_val_score(estimator=clf,
                                 X=X_test,
                                 y=y_test,
                                 cv=5,
                                 n_jobs=1)
        print str(clf) + 'CV accuracy scores: %s' % scores
        print str(clf) + 'CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores))
def gridsearch(clf, X, param_grid_svm):
            X_train, X_test, y_train, y_test = train_test_split(X,
                                                                y,
                                                                test_size=0.25,
                                                                random_state=42)

            gs = GridSearchCV(clf, cv=5, param_grid=param_grid_svm)

            gs.fit(X_train, y_train)
            print str(clf) + 'gridsearch %f' % gs.best_score_
            print gs.best_params_
def precision_recall(model, X):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.25,
                                                        random_state=42)

    svm.fit(X_train, y_train)
    y_score = model.decision_function(X_test)

    from sklearn.metrics import average_precision_score
    average_precision = average_precision_score(y_test, y_score)

    print('Average precision-recall score: {0:0.2f}'.format(
        average_precision))


df_train = pd.read_csv('C:\Users\chadu\PycharmProjects\Titanic-Machine-Learning-from-Disaster-master\data\\train.csv')
df_test = pd.read_csv('C:\Users\chadu\PycharmProjects\Titanic-Machine-Learning-from-Disaster-master\data\\test.csv')

#investigate features
print df_train.head()


print df_train.isnull().sum() #Age, Cabin and Embarked contain NaN values

def preprocessing(df):
    #adjust Cabin feature
    df['Cabin'].fillna('U', inplace=True)  # people without a cabin recorded were probably similar
    df['Cabin'] = df['Cabin'].apply(lambda x: x[0])

    # feature_cat('Cabin', df)

    #adjust Embarked feature
    df['Embarked'].fillna('S', inplace=True)

    import category_encoders as ce
    from sklearn.preprocessing import LabelEncoder

    # encoder = ce.backward_difference.BackwardDifferenceEncoder(cols = ['Sex','Cabin', 'Embarked'])
    # X_en = encoder.fit_transform(df).head()

    lb_make = LabelEncoder()
    df['Cabin'] = lb_make.fit_transform(df['Cabin'])
    df['Embarked'] = lb_make.fit_transform(df['Embarked'])
    # df['Sex'] = lb_make.fit_transform(df['Sex'])

    #would like to try using sklearn CategoricalEncoder, not trouble w availability

    X_dummies = pd.get_dummies(df,columns= ['Sex']) #changed my mind about using dummies...
    #
    # print X_dummies.head()

    # print len(df.columns)
    if len(df.columns) == 12:

        X_drop = X_dummies.drop(['PassengerId', 'Name', 'Ticket','Survived'], axis=1)
        y = X_dummies['Survived']

    else:
        X_drop = X_dummies.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
        y = None


    #
    imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
    X_imr = imr.fit_transform(X_drop)
    # #
    X = pd.DataFrame(X_imr, columns=X_drop.columns)
    #
    X_std = StandardScaler().fit_transform(X)

    return X_std, y


X_std_test, y = preprocessing(df_test)
X_std, y = preprocessing(df_train)

#check feature importance, conclude that cabin and embarked are basically meaningless
# RandomForest(X,y)

forest = RandomForestClassifier(n_estimators = 20)
svm = SVC(kernel = 'rbf')
# testing(forest,X)
# testing(svm,X_std) #0.825 bulk, 0.807 CV

# drop parch, embarked and sibsp which do not appear to have a significant impact on likeihood of survival
# X_reduced = X.drop(['Parch', 'Embarked'], axis=1)


# testing(forest,X_reduced)

#try to improve outcome with gridsearch
param_grid_forest = {"max_depth": [3, None],
              "max_features": [1, 3, 5, 7],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
#svm search parameters
param_grid_svm = [ {'C': [1,1e2,1e3,1e4,1e5,1e6],
                       'gamma': [0.0001,0.0005,0.001,0.005,0.01,0.1]}]

# gridsearch(forest, X_reduced, param_grid_forest )
# gridsearch(svm, X_std, param_grid_svm)
#best score of 0.83 with {'bootstrap': True, 'min_samples_leaf': 3, 'min_samples_split': 2, 'criterion': 'entropy', 'max_features': 3, 'max_depth': None}
#best score of 0.82 with {'C': 1000.0, 'gamma': 0.005}


#test again with these parameters enforced on the classifier
forest_op = RandomForestClassifier(n_estimators = 20, criterion = 'entropy',max_features= 3, max_depth = None, min_samples_split= 2, min_samples_leaf= 3, bootstrap= True)
svm_op = SVC(kernel = 'rbf', C = 1000, gamma = 0.005)
# testing(forest_op,X_reduced)
# testing(svm_op,X_std)

# precision_recall(svm, X_std) #precision-recall stayed the same with optimization


#check results on test set
svm_op.fit(X_std,y)
predict_svm = svm_op.predict(X_std_test)

#export test file results-----------------------------------------------------------------------------------------------

csvfile = "C:\Users\chadu\PycharmProjects\Titanic-Machine-Learning-from-Disaster-master\submissions\\18.3.23\\test_1.csv"

#Assuming res is a flat list
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in predict_svm:
        writer.writerow([val])

#-----------------------------------------------------------------------------------------------------------------------





