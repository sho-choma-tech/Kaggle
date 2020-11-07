import pandas as pd 
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, ShuffleSplit, cross_validate
from sklearn.metrics import log_loss, accuracy_score
from bayes_opt import BayesianOptimization

from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegressionCV, PassiveAggressiveClassifier, RidgeClassifierCV, SGDClassifier, Perceptron
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ------------------------------------------------------
# データの前処理
# ------------------------------------------------------
train = pd.read_csv('../data/input/salary_train.tsv', delimiter='\t')
test = pd.read_csv('../data/input/salary_test.tsv', delimiter='\t')

df_train = train.drop(columns='fnlwgt')
df_test = test.drop(columns='fnlwgt')
df_train['mark'] = 'train'
df_test['mark'] = 'test'

all_data = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)

#前処理
#キャピタルゲイン処理
all_data['capital'] = all_data['capital-gain'] * 1 + all_data['capital-loss'] * (-1)
capital_col = ['capital-gain', 'capital-loss']
for col in capital_col:
    all_data = all_data.drop(columns=col)

#LabelEncoding
le = LabelEncoder()
label_col = ['workclass', 'education', 'occupation', 'relationship', 'race', 'native-country', 'sex', 'marital-status']
for col in label_col:
    all_data[col] = le.fit_transform(all_data[col])

#目的変数エンコーディング
all_data = all_data.replace({'<=50K':0, '>50K':1})


#train data と test dataに分割
train_x = all_data[all_data['mark'] == 'train']
train_y = train_x['Y']
train_x = train_x.drop(columns=['Y', 'mark'])

test_x = all_data[all_data['mark'] == 'test']
test_x = test_x.drop(columns=['Y', 'mark'])


# ------------------------------------------------------
# 学習データ
# ------------------------------------------------------
MLA = [
    #Ensemble Methods
    AdaBoostClassifier()
    ,BaggingClassifier()
    ,ExtraTreeClassifier()
    ,GradientBoostingClassifier()
    ,RandomForestClassifier()

    #Gaussian Processes
    ,GaussianProcessClassifier()

    #GLM
    ,LogisticRegressionCV()
    ,PassiveAggressiveClassifier()
    ,RidgeClassifierCV()
    ,SGDClassifier()
    ,Perceptron()

    #Naive Bayes
    ,BernoulliNB()
    ,GaussianNB()

    #Nearest Neighbor
    ,KNeighborsClassifier()

    #SVM
    ,SVC(probability=True)
    ,NuSVC(probability=True)
    ,LinearSVC()

    #Trees
    ,DecisionTreeClassifier()
    ,ExtraTreeClassifier()

    #Discriminant Analysis
    ,LinearDiscriminantAnalysis()
    ,QuadraticDiscriminantAnalysis()

    #xgboost
    ,XGBClassifier() 

    #lightgbm
    ,LGBMClassifier()
]

cv_split = ShuffleSplit(n_splits=10, test_size=0.3, train_size=0.6, random_state=0)

MLA_columns = [
    'MLA name'
    ,'MLA Parameters'
    ,'MLA Train Accuracy Mean'
    ,'MLA Test Accuracy Mean'
    ,'MLA Test Accuracy 3*STD'
    ,'MLA Time'
]

MLA_compare = pd.DataFrame(columns=MLA_columns)
MLA_predict = train_y.copy()


row_index = 0 
for alg in MLA:
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, "MLA Name"] = MLA_name
    MLA_compare.loc[row_index, "MLA Parameters"] = str(alg.get_params())

    cv_results = cross_validate(alg, train_x, train_y, cv=cv_split, return_train_score=True)

    MLA_compare.loc[row_index, "MLA Time"] = cv_results["fit_time"].mean()
    MLA_compare.loc[row_index, "MLA Train Accuracy Mean"] = cv_results["train_score"].mean()
    MLA_compare.loc[row_index, "MLA Test Accuracy Mean"] = cv_results["test_score"].mean()
    MLA_compare.loc[row_index, "MLA Test Accuracy 3*STD"] = cv_results["test_score"].std()*3

    alg.fit(train_x, train_y)
    MLA_predict[MLA_name] = alg.predict(test_x)

    row_index += 1

MLA_compare