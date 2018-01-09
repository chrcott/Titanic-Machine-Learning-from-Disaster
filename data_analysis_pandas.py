import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_train = pd.read_csv('C:\Users\chaduda\Kaggle\Titanic Machine Learning from Disaster\Data\\train.csv')

df_train.fillna(-999,inplace=True)

def feature_cat(feature, df):
    feature_count = df.groupby(feature).PassengerId.count()
    feature_sum = df.groupby(feature).Survived.sum()

    percent_survived = feature_sum/feature_count*100

    percent_survived.plot(kind='bar')

    plt.show()

def feature_dis(feature,df,bin_max,bin_number):

    bins = np.linspace(0,bin_max,bin_number)

    feature_count = df.groupby(pd.cut(df[feature],bins)).PassengerId.count()
    feature_sum = df.groupby(pd.cut(df[feature],bins)).Survived.sum()

    percent_survived = feature_sum / feature_count * 100

    # df[feature].plot.box(vert=False) #have a look at the distribution
    # percent_survived.plot(kind='bar')
    feature_count.plot(kind='bar')


    plt.show()

# feature_cat('Sex', df_train)
# feature_cat('Pclass', df_train)
# feature_cat('SibSp', df_train)
# feature_cat('Parch', df_train)
# feature_dis('Age',df_train,80,9)
feature_dis('Fare',df_train,300,30) #only one person in the highest paying bracket

