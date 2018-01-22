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

X_train, y_train = df_dummies.iloc[:,1:].values, df_dummies.iloc[:,0].values

imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
X_train_imr = imr.fit_transform(X_train)


stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train_imr) #dont want to standardize categorial values

# X_train = pd.DataFrame(X_train_std, columns=X.columns)

def pca():
    pca = PCA(n_components=2)
    lr = LogisticRegression()
    X_train_pca = pca.fit_transform(X_train_std)
    # X_test_pca = pca.transform(X_test_std)
    lr.fit(X_train_pca, y_train)
    plot_decision_regions(X_train_pca, y_train, classifier=lr)
    plt.show()

RandomForest(X_train_std,y_train)