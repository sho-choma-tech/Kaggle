# ------------------------------------------------------
# 時系列hold-out法
# ------------------------------------------------------



# ------------------------------------------------------
# lightgbmの実装
# ------------------------------------------------------
import pandas as pd 
import xgboost as xgb 
import lightgbm as lgb 


train_x_ = train_x.copy()
train_y_ = train_y.copy()
test_x_ = test_x.copy()
test_y_ = test_y.copy()


lgb_train = lgb.Dataset(train_x_, label=train_y_)
lgb_eval  = lgb.Dataset(test_x_, label=test_y_)

params_xgb = {
    'silent':1
    , 'random_state':71
    , 'objective':'regression_l1'
    , 'max_depth':5
    , 'early_stopping_rounds':500
    }

num_round = 20 

model_lgb = lgb.train(
    params_xgb
    ,lgb_train
    ,valid_sets=lgb_eval
    ,num_boost_round=10000
    ,early_stopping_rounds=100
    ,verbose_eval=50
)

#print(f'mean absolute score:{score:.4f}')



# ------------------------------------------------------
# submit用データ作成
# ------------------------------------------------------

test_df_merge = test_df[['id', 'target']]

pred_data = pd.merge(test_x_, test_df_merge, on='id', how='left')
pred_data = pred_data[pred_data['target'] == 1]

#予測値を出力
va_pred = model_lgb.predict(pred_data)

#欠損値 target = 1 を予測
test_df_merge = test_df_merge[test_df_merge['target'] == 1]

idx = test_df_merge['id'].reset_index(drop=True)
val = pd.Series(va_pred).reset_index(drop=True)
print('The amount of index counts are ', len(idx))
print('The amount of values counts are',  len(val))

#submission dataの提出
submit_df = pd.concat([idx, val], axis=1)
submit_df.to_csv('./data/output/(2020-12-15_lgb_lineStation).csv', header=False, index=False)