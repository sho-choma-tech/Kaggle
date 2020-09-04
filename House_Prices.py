#!/usr/bin/env python
# coding: utf-8

# In[161]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import math 
import warnings
warnings.filterwarnings('ignore')

from lightgbm import LGBMRegressor
from xgboost import XGBRFRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error


# In[2]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

df_train = train.copy()
df_test = test.copy()


# In[3]:


train.head()


# In[4]:


test.head()


# In[6]:


df_train['train'] = 1
df_test['train'] = 0 

df_data = pd.concat([df_train, df_test], axis=0, sort=False)


# In[29]:


NA = [(i, df_data[i].isna().mean() * 100) for i in df_data]
NA = pd.DataFrame(NA, columns=['Feature Names', 'Percentage'])
NA.sort_values('Percentage' ,ascending=False).head(10)


# In[31]:


df_data = df_data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis=1)


# In[33]:


object_columns_df = df_data.select_dtypes(include='object')
numerical_columns_df = df_data.select_dtypes(exclude='object')


# In[36]:


object_columns_df.head()


# In[50]:


columns_null_count = [(col, object_columns_df[col].isnull().sum()) for col in object_columns_df.columns]
columns_null_count = pd.DataFrame(columns_null_count, columns=['Feature Name', 'count'])
columns_null_count[columns_null_count['count'] != 0].sort_values('count', ascending=False)


# In[65]:


columns_none = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'GarageType', 'GarageFinish', 'FireplaceQu', 'GarageCond']
object_columns_df[columns_none] = object_columns_df[columns_none].fillna('None')


# In[67]:


columns_low_none = ['MSZoning', 'Functional', 'Utilities', 'KitchenQual', 'Electrical', 'Exterior2nd', 'Exterior1st', 'SaleType']
object_columns_df[columns_low_none] = object_columns_df[columns_low_none].fillna(object_columns_df[columns_low_none].mode().iloc[0])


# In[63]:


numerical_null_count = [(col, numerical_columns_df[col].isnull().sum()) for col in numerical_columns_df.columns]
numerical_null_count = pd.DataFrame(numerical_null_count, columns=['Feature Name', 'count'])
numerical_null_count[numerical_null_count['count'] != 0].sort_values('count',ascending=False)


# In[74]:


numerical_columns_df['LotFrontage'] = numerical_columns_df['LotFrontage'].fillna(numerical_columns_df['LotFrontage'].median())
numerical_columns_df['GarageYrBlt'] = numerical_columns_df['GarageYrBlt'].fillna(numerical_columns_df['YrSold'] - (numerical_columns_df['YrSold'] - numerical_columns_df['YearBuilt']).median())
numerical_columns_df = numerical_columns_df.fillna(0)


# In[112]:


#fig, axes = plt.subplots(3,4, figsize=(16,12))
#for i in range(3):
#    for j in range(4):
#        for col in object_columns_df.columns:
#            axes[i][j].bar(columns_null_count1, height= data=object_columns_df)


# In[113]:


#columns_null_count1 = [(col, object_columns_df[col].value_counts()) for col in object_columns_df.columns]
#columns_null_count1


# Exploring Data Analytics

# In[83]:


plt.figure(figsize=(10,7))
sns.countplot(x = 'Utilities', data = object_columns_df)
plt.grid(True)
object_columns_df['Utilities'].value_counts()


# In[84]:


plt.figure(figsize=(10,7))
sns.countplot(x = 'Street', data = object_columns_df)
plt.grid(True)
object_columns_df['Street'].value_counts()


# In[85]:


plt.figure(figsize=(10,7))
sns.countplot(x = 'Condition2', data = object_columns_df)
plt.grid(True)
object_columns_df['Condition2'].value_counts()


# In[114]:


plt.figure(figsize=(10,7))
sns.countplot(x = 'RoofMatl', data = object_columns_df)
plt.grid(True)
object_columns_df['RoofMatl'].value_counts()


# In[115]:


plt.figure(figsize=(10,7))
sns.countplot(x = 'Heating', data = object_columns_df)
plt.grid(True)
object_columns_df['Heating'].value_counts()


# In[116]:


object_columns_df = object_columns_df.drop(['Utilities', 'Street', 'Condition2', 'RoofMatl', 'Heating'], axis=1)


# create the new feature

# In[118]:


numerical_columns_df[['YrSold', 'YearBuilt']]


# In[120]:


numerical_columns_df['Age_House'] = (numerical_columns_df['YrSold'] - numerical_columns_df['YearBuilt'])
numerical_columns_df['Age_House'].describe()


# In[121]:


numerical_columns_df[numerical_columns_df.Age_House < 0]


# In[122]:


numerical_columns_df.head(10)


# In[123]:


numerical_columns_df.loc[numerical_columns_df['YearBuilt'] > numerical_columns_df['YrSold'], 'YrSold'] = 2009
numerical_columns_df['Age_House'] = (numerical_columns_df['YrSold'] - numerical_columns_df['YearBuilt'])
numerical_columns_df['Age_House'].describe()


# In[125]:


numerical_columns_df['TotalbsmtBath'] = numerical_columns_df['BsmtFullBath'] + numerical_columns_df['BsmtFullBath'] * 0.5 
numerical_columns_df['TotalBath'] = numerical_columns_df['FullBath'] + numerical_columns_df['HalfBath'] * 0.5 
numerical_columns_df['TotalSa'] = numerical_columns_df['TotalBsmtSF'] + numerical_columns_df['1stFlrSF'] + numerical_columns_df['2ndFlrSF']


# In[129]:


object_columns_df.head()


# In[130]:


bin_map = {
    'Fa':1, 'Ex':4, 'Gd':3, 'TA':2, 'Po':1, 'None':0
    , 'Y':1, 'N':0 
    , 'Reg':3, 'IR1':2, 'IR2':1, 'IR3':0
    , 'No':2, 'Mn':2, 'Av':3, 'Gd':4
    , 'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6
}

PaveDrive = {
    'N':0, 'P':1, 'Y':2
}

object_columns_df['LotShape'] = object_columns_df['LotShape'].map(bin_map)
object_columns_df['ExterQual'] = object_columns_df['ExterQual'].map(bin_map)
object_columns_df['ExterCond'] = object_columns_df['ExterCond'].map(bin_map)
object_columns_df['BsmtCond'] = object_columns_df['BsmtCond'].map(bin_map)
object_columns_df['BsmtQual'] = object_columns_df['BsmtQual'].map(bin_map)
object_columns_df['HeatingQC'] = object_columns_df['HeatingQC'].map(bin_map)
object_columns_df['KitchenQual'] = object_columns_df['KitchenQual'].map(bin_map)
object_columns_df['FireplaceQu'] = object_columns_df['FireplaceQu'].map(bin_map)
object_columns_df['GarageQual'] = object_columns_df['GarageQual'].map(bin_map)
object_columns_df['GarageCond'] = object_columns_df['GarageCond'].map(bin_map)
object_columns_df['CentralAir'] = object_columns_df['CentralAir'].map(bin_map)
object_columns_df['BsmtExposure'] = object_columns_df['BsmtExposure'].map(bin_map)
object_columns_df['BsmtFinType1'] = object_columns_df['BsmtFinType1'].map(bin_map)
object_columns_df['BsmtFinType2'] = object_columns_df['BsmtFinType2'].map(bin_map)
object_columns_df['PavedDrive'] = object_columns_df['PavedDrive'].map(PaveDrive)


# In[134]:


new_object_columns_df = object_columns_df.select_dtypes(include='object')
object_columns_df = pd.get_dummies(object_columns_df, columns=new_object_columns_df.columns)


# In[138]:


model_data = pd.concat([object_columns_df, numerical_columns_df], axis=1, sort=False)
model_data.head()


# In[139]:


model_data = model_data.drop(['Id'],axis=1)


# In[140]:


model_train = model_data[model_data['train'] == 1]
model_train = model_train.drop(['train'], axis=1)

model_test = model_data[model_data['train'] == 0]
model_test = model_test.drop(['SalePrice'], axis=1)
model_test = model_test.drop(['train'], axis=1)


# In[143]:


target = model_train['SalePrice']
model_train = model_train.drop(['SalePrice'], axis=1)


# Modeling

# In[145]:


X_train, X_test, y_train, y_test = train_test_split(model_train, target, test_size=0.33, random_state=0)


# In[157]:


xgb = XGBRFRegressor(colsample_bynode=1, colsample_bytree=0.6, learning_rate=0.01, max_delta=4
                     , min_child_weight=1.5, n_estimators=2400, reg_alpha=0.6, reg_lambda=0.6)
lgbm = LGBMRegressor(objective='regression', num_leaves=4, learning_rate=0.01, n_estimators=12000)


# In[158]:


xgb.fit(X_train, y_train)
lgbm.fit(X_train, y_train, eval_metric='rmse')


# In[162]:


predict1 = xgb.predict(X_test)
predict2 = lgbm.predict(X_test)


# In[164]:


print('Root Mean Square Error test = ' + str(math.sqrt(mean_squared_error(y_test, predict1))))
print('Root Mean Square Erroe test = ' + str(math.sqrt(mean_squared_error(y_test, predict2))))


# In[165]:


predcict3 = lgbm.predict(model_test)
predcict4 = xgb.predict(model_test)
predict_y = (predcict3*0.45 + predcict4*0.55)


# In[167]:


submission = pd.DataFrame({
    'Id':test['Id']
    , 'SalePrice':predict_y
    })
submission.to_csv('submission.csv', index=False)


# In[ ]:




