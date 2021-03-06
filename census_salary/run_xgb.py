import pandas as pd 
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import log_loss, accuracy_score
import lightgbm as lgb 
import catboost as cat  
import xgboost as xgb 


train = pd.read_csv('./data/input/salary_train.tsv', delimiter='\t')
test = pd.read_csv('./data/input/salary_test.tsv', delimiter='\t')

df_train = train.drop(columns='fnlwgt')
df_test = test.drop(columns='fnlwgt')
df_train['mark'] = 'train'
df_test['mark'] = 'test'

all_data = pd.concat([df_train, df_test], axis=0)

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



# ------------------------------------------------------
# KFold分割
# ------------------------------------------------------
train_x = all_data[all_data['mark'] == 'train']
train_y = train_x['Y']
train_x = train_x.drop(columns=['Y', 'mark'])

test_x = all_data[all_data['mark'] == 'test']
test_x = test_x.drop(columns=['Y', 'mark'])



kf = KFold(n_splits=4, shuffle=True, random_state=71)
tr_idx, va_idx = list(kf.split(train_x))[0]

tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

# ------------------------------------------------------
# xgboostの実装
# ------------------------------------------------------

scores = []

xgb_params = {
    'objective':'binary:logistic'
    , 'eval_metrics':'logloss'
}
num_round = 100 

dtrain = xgb.DMatrix(tr_x, label=tr_y)
dvalid = xgb.DMatrix(va_x, label=va_y)
dtest = xgb.DMatrix(test_x)

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
bst = xgb.train(xgb_params, dtrain, num_boost_round=num_round, evals=watchlist)

va_pred = bst.predict(dvalid)
y_pred = np.where(va_pred > 0.5, 1, 0)
acc_score = accuracy_score(va_y, y_pred)
print(f'accuracy:{acc_score:.4f}')

pred = bst.predict(dtest)
pred_target = np.where(pred >= 0.5, '>50K', '<=50K')


pred_id = test_x['id']
pred_target = pd.DataFrame(pred_target, columns=['Y'])
submission_data = pd.concat([pred_id, pred_target], axis=1)

submission_data.to_csv('./data/input/(2020-11-7)_xgb.csv', index=False)