import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import KernelPCA, PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score

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

def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    # for idx, cl in e:
    #     numerate(np.unique(y)):
    #     plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
    #             alpha=0.8, c=cmap(idx),
    #             marker=markers[idx], label=cl)
def plot_decision_regions_compare(X, y, classifier,
 test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot all samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
    plt.scatter(X_test[:, 0], X_test[:, 1], c='',
                alpha=1.0, linewidths=1, marker='o',
                s=55, label='test set')

def LDA():
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_train_lda = lda.fit_transform(X_train_std, y_train)

    plt.scatter(X_train_lda[y_train == 0, 0], X_train_lda[y_train == 0, 1],
                color='red', marker='^', alpha=0.5)

    plt.scatter(X_train_lda[y_train == 1, 0], X_train_lda[y_train == 1, 1],
                color='blue', marker='o', alpha=0.5)

    # lr = LogisticRegression()
    # lr = lr.fit(X_train_lda, y_train)
    # plot_decision_regions(X_train_lda, y_train, classifier=lr)
    # plt.xlabel('LD 1')
    # plt.ylabel('LD 2')
    # plt.legend(loc='lower left')
    # plt.show()


def kernalpca():
    scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)

    X_skernpca = scikit_kpca.fit_transform(X_train_imr)

    plt.scatter(X_skernpca[y_train == 0, 0], X_skernpca[y_train == 0, 1],
                color='red', marker='^', alpha=0.5)

    plt.scatter(X_skernpca[y_train == 1, 0], X_skernpca[y_train == 1, 1],
                color='blue', marker='o', alpha=0.5)
    plt.show()


'C:\Users\chaduda\PycharmProjects\Kaggle\Titanic_Machine_Learning_from_Disaster\machine_learning.py'

df = pd.read_csv('C:\Users\chaduda\Kaggle\Titanic Machine Learning from Disaster\Data\\train.csv')

df.isnull().sum() #missing values for Age and Cabin columns

df_drop = df.drop(['Name','Cabin','Ticket','PassengerId'], axis=1)

df_dummies = pd.get_dummies(df_drop,columns= ['Sex','Embarked'])

X = df_dummies.iloc[:,1:].values
y_train = df_dummies.iloc[:, 0].values

imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
X_train = imr.fit_transform(X)

# stdsc = StandardScaler()
# X_imr_std = stdsc.fit_transform(X_imr)

# X_train, X_test, y_train, y_test = train_test_split(X_imr,
#                                                     df_dummies.iloc[:, 0].values,
#                                                     test_size = 0.4,
#                                                    random_state=0)

pipe_svm = Pipeline([('scl', StandardScaler()),
                     ('svm', SVC(kernel = 'rbf', C = 1.0))])
pipe_svm.fit(X_train, y_train)
# print 'pipe svm score:' +str(pipe_svm.score(X_test,y_test)) #I have no test data

scores = cross_val_score(estimator=pipe_svm,
                         X = X_train,
                         y=y_train,
                         cv=10,
                         n_jobs=1)
print 'CV accuracy scores: %s' %scores
print 'CV accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores))

# X_train = pd.DataFrame(X_train_std, columns=X.columns)


def pca():
    pca = PCA(n_components=2)
    lr = LogisticRegression()
    X_train_pca = pca.fit_transform(X_train_std)
    # X_test_pca = pca.transform(X_test_std)
    lr.fit(X_train_pca, y_train)
    plot_decision_regions(X_train_pca, y_train, classifier=lr)
    plt.show()



# RandomForest(X_train,y_train)

# tree = DecisionTreeClassifier(criterion='entropy',
#                               max_depth=3, random_state=0)
# tree.fit(X_train,y_train)
# X_combined = np.vstack((X_train, X_test))
# y_combined = np.hstack((y_train, y_test))
# plot_decision_regions_compare(X_train, y_train, classifier = tree,
#                               test_idx=range(105,150))
# # plt.show()

#classification methods:

#nearest neighbors
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train, y_train)
# print knn.score(X_test, y_test)

#logistic regression
# lr = LogisticRegression(C=100, random_state=0)
# lr.fit(X_train, y_train)
# print lr.score(X_test, y_test)

#support vector machines
# svm = SVC(kernel = 'poly', C=1.0, random_state=0)
# svm.fit(X_train, y_train)
# print svm.score(X_test, y_test)

# scores = cross_val_score(estimator =)

#Decision tree
# tree = DecisionTreeClassifier(criterion='entropy',
#                               max_depth=4, random_state=0)
# tree.fit(X_train,y_train)
# print "tree score:" +str(tree.score(X_test,y_test))

#random forest
# forest = RandomForestClassifier(criterion = 'entropy',
#                                 n_estimators = 10,
#                                 random_state = 1,
#                                 n_jobs = 2)
# forest.fit(X_train, y_train)
# print forest.score(X_test, y_test)

#initiate gridsearch to compare machine learning algorithms

# param_range = [0.01,0.1,1.0]
# param_grid_svm = [{'C': param_range,
#                    'kernel': ['rbf']},
#                   {'C': param_range,
#                    'gamma': param_range,
#                    'kernel': ['rbf']}]
# gs = GridSearchCV(estimator = SVC(random_state = 1),
#                   param_grid = param_grid_svm,
#                   scoring = 'accuracy',
#                   cv=10)
# gs.fit(X_train, y_train)
# print 'svm gridsearch %f' %gs.best_score_
# print gs.best_params_


# param_range_tree = np.linspace(1,10,10)
# param_grid_tree = [{'max_depth': param_range_tree,
#                    'criterion': ['gini']},
#                   {'max_depth': param_range_tree,
#                    'criterion': ['entropy']}]
# gs = gs_tree = GridSearchCV(estimator = DecisionTreeClassifier(random_state=1),
#                   param_grid = param_grid_tree,
#                   scoring = 'accuracy',
#                   cv=10)
# gs.fit(X_train, y_train)
# print 'tree gridsearch %f' %gs.best_score_
# print gs.best_params
#
# param_range_forest = np.linspace(1,50,10, dtype = int)
# param_grid_forest = [{'n_estimators': param_range_forest,
#                    'criterion': ['gini']},
#                   {'n_estimators': param_range_forest,
#                    'criterion': ['entropy']}]
# gs = gs_tree = GridSearchCV(estimator = RandomForestClassifier(random_state=1),
#                   param_grid = param_grid_forest,
#                   scoring = 'accuracy',
#                   cv=10)
# gs.fit(X_train, y_train)
# print 'forest gridsearch %f' %gs.best_score_
# print gs.best_params_

# param_range_knn = np.linspace(1,20,19, dtype = int)
# param_grid_knn = [{'n_neighbors': param_range_knn,
#                    'algorithm': ['ball_tree']},
#                   {'n_neighbors': param_range_knn,
#                    'algorithm': ['kd_tree']},
#                      {'n_neighbors': param_range_knn,
#                       'algorithm': ['brute']}]
# gs = gs_knn = GridSearchCV(estimator = KNeighborsClassifier(),
#                   param_grid = param_grid_knn,
#                   scoring = 'accuracy',
#                   cv=10)
# gs.fit(X_train, y_train)
# print 'knn gridsearch %f' %gs.best_score_
# print gs.best_params_

# param_range_lr = [0.01,0.1,1]
# param_grid_lr = [{'C': param_range_lr},
#                   {'C': param_range_lr,
#                    'penalty': ['l1']}]
# gs = GridSearchCV(estimator = LogisticRegression(random_state=1),
#                   param_grid = param_grid_lr,
#                   scoring = 'accuracy',
#                   cv=10)
# gs.fit(X_train, y_train)
# print 'lr gridsearch %f' %gs.best_score_
# print gs.best_params_

