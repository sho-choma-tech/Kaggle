import pandas as pd 
import numpy as np 
import math

import lightgbm as lgb
import xgboost as xgb
import catboost as cat


from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')


# ------------------------------------------------------
# 説明変数と目的変数の宣言
# ------------------------------------------------------

df_data = pd.concat([obj ect_df, numerical_df], axis=1, sort=False)
df_data = df_data.drop(columns=['index'])

df_train = df_data[df_data['mark'] == 1]
df_train = df_train.drop(columns=['mark'])

df_test = df_data[df_data['mark'] == 0]
test_x = df_test.drop(columns=['mark', 'SalePrice'])

train_y = df_train['SalePrice']
train_x = df_train.drop(columns=['SalePrice'])


# ------------------------------------------------------
# Validation_Function
# ------------------------------------------------------
def rmsle_cv(model):
    kf = KFold(n_splits=5, shuffle=True, random_state=42).get_n_splits(train_x.values)
    rmse = np.sqrt(-cross_val_score(model, train_x.values, train_y, scoring='neg_mean_squared_error', cv=kf))
    return rmse


# RobustScalerの特徴
# データポイントが、中央値が0になり、四分位範囲（interquartile range、IQR）が1になるように移動・スケール変換する。
# 5個の要素を持つデータポイントでは、25%分位数は2番目の要素になる。
# 平均と分散を使った移動・スケール変換に比べて、外れ値の影響を受けにくい。


# Lasso model 
lasso_ = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
score_lasso = rmsle_cv(lasso_)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score_lasso.mean(), score_lasso.std()))


#Elastic Net Regression model 
ENet_ = make_pipeline(RobustScaler(), ElasticNet(alpha=0.005, l1_ratio=0.9, random_state=3))
score_ENet = rmsle_cv(ENet_)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score_ENet.mean(), score_ENet.std()))


# Gauss Kernel model 
# 周期的に変動するデータは線形モデルと相性が悪い → ガウスカーネルモデル
KRR_ = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
score_KRR = rmsle_cv(KRR_)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score_KRR.mean(), score_KRR.std()))


# 
GradientBoost_ = GradientBoostingRegressor(
    n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt'
    , min_samples_leaf=15, min_samples_split=10, loss='huber', random_state=5
)
score_Gradient = rmsle_cv(GradientBoost_)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score_Gradient.mean(), score_Gradient.std()))


# Xgboost model 
xgb_ = xgb.XGBRegressor(
    colsample_bytree=0.4603, gamma=0.0468, learning_rate=0.05, max_depth=3
    , min_child_weight=1.7817, n_estimators=2200, reg_alpha=0.4640, reg_lambda=0.8571
    , subsample=0.5213, silent=1, random_state=7, nthread=-1
)
xgb_score = rmsle_cv(xgb_)
print('\nXGBoost score:{:.4f}({:.4f})\n'.format(xgb_score.mean(), xgb_score.std()))


# Lightgbm model 
lgb_ = lgb.LGBMRegressor(
    objective='regression', num_leaves=5, learning_rate=0.05, n_estimators=720,
    max_bin = 55, bagging_fraction = 0.8, bagging_freq = 5, feature_fraction = 0.2319,
    feature_fraction_seed=9, bagging_seed=9, min_data_in_leaf=6, min_sum_hessian_in_leaf=11
)
lgb_score = rmsle_cv(lgb_)
print("\nLightGBM score:{:.4f}({:.4f})\n",format(lgb_score.mean(), lgb_score.std()))



# ------------------------------------------------------
# Stacking model
# ------------------------------------------------------
class AverageModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models 

    # We define clones of the original models to fit then data in 
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models 
        for model in self.models_:
            model.fit(X, y)

        return self

    # Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])

        return np.mean(predictions, axis=1)

averaged_models = AverageModels(models=(ENet_, GradientBoost_, KRR_, lasso_))
score = rmsle_cv(averaged_models)



class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    # Do the predictions of all base models on the test data and use the averaged predictions as 
    # meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)




stacked_averaged_models = StackingAveragedModels(
    base_models = (ENet_, GradientBoost_, KRR_), meta_model = lasso_)

score = rmsle_cv(stacked_averaged_models)
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))




# ------------------------------------------------------
# Ensembling StackedRegressor, XGBoost and LightGBM
# ------------------------------------------------------


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
    
# Stacked model 
stacked_averaged_models.fit(train_x.values, train_y)
stacked_train_pred = stacked_averaged_models.predict(train_x.values)
stacked_pred = stacked_averaged_models.predict(test_x.values)
print(rmsle(train_y, stacked_train_pred))


# XGBoost model 
xgb_.fit(train_x, train_y)
xgb_train_pred = xgb_.predict(train_x)
xgb_pred = xgb_.predict(test_x)
print(rmsle(train_y, xgb_train_pred))


#LightBGM model 
lgb_.fit(train_x, train_y)
lgb_train_pred = lgb_.predict(train_x)
lgb_pred = lgb_.predict(test_x)
print(rmsle(train_y, lgb_train_pred))


#RMSE on the entire Train data when averaging
print('RMSLE score on train data:')
print(rmsle(train_y,stacked_train_pred*0.70 + xgb_train_pred*0.15 + lgb_train_pred*0.15 ))


# the most accurate model score 
ensemble = stacked_pred*0.80 + lgb_pred*0.20


# the model which tests next evaluation 
ensemble4 = stacked_pred*0.90 + lgb_pred*0.10



# submission data を作成
submission_data = pd.DataFrame({
    'index':test['index']
    , 'SalePrice':ensemble
    }
)

filepath = './data/output/(2020_01_04)_.csv'
submission_data.to_csv(filepath, header=False, index=False)
