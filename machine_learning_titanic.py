import pandas as pd
import numpy as np




from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
import csv
import matplotlib.pyplot as plt

def RandomForest(X_train,y_train):
    forest = RandomForestClassifier(n_estimators=20,
                                    random_state = 42,
                                    n_jobs=-1)
    forest.fit(X_train,y_train)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)

    indices = np.argsort(importances)[::-1]
    feat_labels = X_train.columns

    for f in range(X_train.shape[1]):
             print '%2d %-*s %f' % (f+1,30,
                                    feat_labels[indices[f]],
                                    importances[indices[f]])
    # print feature
    # print importance
    plt.title('Random Forest Classifier')
    plt.xlabel('Feature')
    plt.ylabel('Feature Importance')
    plt.bar(range(X_train.shape[1]), importances[indices],
            color="b", yerr=std[indices], align="center")
    plt.xticks(range(len(feat_labels)), feat_labels[indices])
    # plt.bar(x,height=importance,color='g')
    plt.show()
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
                                                            test_size=0.30,
                                                            random_state=42)

        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        # print str(clf) + '\n accuracy score: %.3f' % score

        scores = cross_val_score(estimator=clf,
                                 X=X_test,
                                 y=y_test,
                                 cv=5,
                                 n_jobs=1)
        # print str(clf) + 'CV accuracy scores: %s' % scores
        print str(clf) + '\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores))
def gridsearch(clf, X, param_grid):
            X_train, X_test, y_train, y_test = train_test_split(X,
                                                                y,
                                                                test_size=0.30,
                                                                random_state=42)

            gs = GridSearchCV(clf, cv=5, param_grid=param_grid)

            gs.fit(X_train, y_train)
            print str(clf) + '\nbest score: %f' % gs.best_score_
            print 'best hyperparaters: C={C}, gamma={gamma}'.format(**gs.best_params_)
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
# print df_train.head()


print df_train.isnull().sum() #Age, Cabin and Embarked contain NaN values

def preprocessing(df):
    #adjust Cabin feature
    df['Cabin'].fillna('U', inplace=True)  # people without a cabin recorded were probably similar
    df['Cabin'] = df['Cabin'].apply(lambda x: x[0])

    # feature_cat('Cabin', df)

    #adjust Embarked feature
    df['Embarked'].fillna('S', inplace=True)

    #adjust Name feature
    f = lambda x: x.split(',')[1].split('.')[0]
    df['Name'] = df['Name'].apply(f)


    import category_encoders as ce
    from sklearn.preprocessing import LabelEncoder

    # encoder = ce.backward_difference.BackwardDifferenceEncoder(cols = ['Sex','Cabin', 'Embarked'])
    # X_en = encoder.fit_transform(df).head()

    lb_make = LabelEncoder()
    df['Cabin'] = lb_make.fit_transform(df['Cabin'])
    df['Embarked'] = lb_make.fit_transform(df['Embarked'])
    df['Name'] = lb_make.fit_transform(df['Name'])
    # df['Sex'] = lb_make.fit_transform(df['Sex'])

    print df.head()
    # print 'name %f' %(len(df['Name'].unique()))
    # print 'cabin %f' % (len(df['Cabin'].unique()))
    # print 'em %f' % (len(df['Embarked'].unique()))


    #would like to try using sklearn CategoricalEncoder, not trouble w availability

    X_dummies = pd.get_dummies(df,columns= ['Sex']) #changed my mind about using dummies...
    #
    # X_dummies = df

    # print len(df.columns)
    if len(df.columns) == 12:

        X_drop = X_dummies.drop(['PassengerId', 'Ticket','Survived'], axis=1)
        y = X_dummies['Survived']

    else:
        X_drop = X_dummies.drop(['PassengerId','Ticket'], axis=1)
        y = None


    #
    imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
    X_imr = imr.fit_transform(X_drop)
    # #
    X = pd.DataFrame(X_imr, columns=X_drop.columns)
    #

    # RandomForest(X, y)


    # X_std = StandardScaler().fit_transform(X)

    return X, y
#removed std

X_test, y = preprocessing(df_test)
X_train, y = preprocessing(df_train)

X_std_test = StandardScaler().fit_transform(X_test)
X_std_train = StandardScaler().fit_transform(X_train)


#check feature importances
# RandomForest(X_train,y)

#call estimators with no hyper-parameters set
svm = SVC()
knn = KNeighborsClassifier()
lr = LogisticRegression()

#run testing funtion w standardized data
# testing(svm,X_std_train)
# testing(knn,X_std_train)
# testing(lr,X_std_train)

#try to improve outcome with gridsearch
#feel like using gridsearch doesn't improve my results enough for bother
#setting hyperparameter values and ranges to be investigated
#svm search parameters
param_grid_svm = [ {'C': [1,1e2,1e3,1e4],
                       'gamma': [0.0001,0.001,0.01,0.1]}]
#knn search parameters
param_grid_knn = [ {'n_neighbors': [3,4,5,6,7,8,9,10]}]

#calling gridsearch
# gridsearch(svm, X_std_train, param_grid_svm)
# gridsearch(knn, X_std_train, param_grid_knn)


#test again, this time enforcing the hyperparameters generated from the gridsearch
# svm_op = SVC(C=1.0, gamma = 0.1, kernel = 'rbf')
# testing(svm_op,X_std_train)

#check results on test set
# svm.fit(X_std_train,y)
# predict_svm = svm.predict(X_std_test)

# lr.fit(X_std_train,y)
# predict_lr = lr.predict(X_std_test)

# knn.fit(X_std_train,y)
# predict_knn = knn.predict(X_std_test)


#export test file results-----------------------------------------------------------------------------------------------

csvfile = "C:\Users\chadu\PycharmProjects\Titanic-Machine-Learning-from-Disaster-master\submissions\\18.5.24\\sub_3.csv"

# Assuming res is a flat list
# with open(csvfile, "w") as output:
#     writer = csv.writer(output, lineterminator='\n')
#     for val in predict_svm:
#         writer.writerow([val])

#-----------------------------------------------------------------------------------------------------------------------





