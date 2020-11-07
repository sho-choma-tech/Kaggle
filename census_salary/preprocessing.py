import pandas as pd 
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import log_loss, accuracy_score
from bayes_opt import BayesianOptimization
import lightgbm as lgb 
import catboost as cat  
import xgboost as xgb 

train = pd.read_csv('../data/salary_train.tsv', delimiter='\t')
test = pd.read_csv('../data/salary_test.tsv', delimiter='\t')

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

