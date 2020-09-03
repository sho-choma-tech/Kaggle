

import numpy as np
import os 
import pandas as pd
import scipy as sp 
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from google.colab import drive 
drive.mount('/content/drive')

df_train = pd.read_csv('drive/My Drive/Kaggle/Titanic/train.csv')
df_test = pd.read_csv('drive/My Drive/Kaggle/Titanic/test.csv')

df_train.head()

df_test.head()

df_train.info()
print('------------------')
df_test.info()

for col in df_train.columns:
  print(col, round(df_train[col].isnull().sum() / len(df_train) * 100, 2), '%')
print('---------------------------------')
for col in df_test.columns:
  print(col, round(df_test[col].isnull().sum() / len(df_test) * 100, 2), '%')

df_train.describe().T

plt.figure(figsize=(10,7))
sns.countplot(x = 'Survived', data = df_train)
plt.grid(True)

plt.figure(figsize=(10,7))
sns.countplot(x = 'Sex', data = df_train)
plt.grid(True)

plt.figure(figsize=(15,7))
sns.countplot(x = 'Pclass', data = df_train)
plt.grid(True)

plt.figure(figsize=(15,7))
sns.countplot(x = 'Embarked', data = df_train)
plt.grid(True)

plt.figure(figsize=(15,7))
sns.countplot(x = 'SibSp', data = df_train)
plt.grid(True)

plt.figure(figsize=(15,7))
sns.countplot(x = 'Parch', data = df_train)
plt.grid(True)

plt.figure(figsize=(12,8))
sns.distplot(df_train.Age, hist = True, kde=True)
plt.show()

plt.figure(figsize=(12,8))
sns.distplot(df_train.Fare, hist = True, kde=True, color='g')

plt.figure(figsize=(15, 7))
sns.countplot(df_train.Sex, hue=df_train.Survived, palette='pastel')
plt.grid(True)

plt.figure(figsize=(15, 7))
sns.countplot(df_train.Pclass, hue=df_train.Survived, palette='pastel')
plt.grid(True)

"""Missing Values"""

data = pd.concat([df_train, df_test])
data.head()

for i in data.columns:
  print(i, round(data[i].isnull().sum() / len(data) * 100, 2), '%')

data.isnull().sum()

data[data['Embarked'].isnull()]

plt.figure(figsize=(15,7))
sns.catplot(x = 'Embarked', y = 'Fare', data = data, kind='box')
plt.grid(True)

data['Embarked'] = data.Embarked.fillna('C')

data[data.Fare.isnull()]
data['Fare'] = data.Fare.fillna(data.Fare.median())

plt.figure(figsize=(15,7))
sns.distplot(data.Age, kde=True)
plt.grid(True)

print('Median : ', data.Age.median())
print('Mean : ', data.Age.mean())

data['Age'] = data.Age.fillna(data.Age.median())

data.head()

data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

data.head()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])
data['Embarked'] = le.fit_transform(data['Embarked'])
data.head()

train = data.iloc[0: (data.shape[0] - df_test.shape[0])]
test = data.iloc[df_train.shape[0] : ]

train.head()

test.tail()

X = train.drop('Survived', axis=1)
y = train.Survived

"""Random Forest"""

rf = RandomForestClassifier()
rf_params = {
    'n_estimators' : [400,500,600,700]
    , 'max_features' : [5,6,7,8,9,10]
    , 'min_samples_split' : [5,6,7,8,9,10]
}

rf_cv_model = GridSearchCV(rf, rf_params, cv=21, n_jobs=-1, verbose=1).fit(X, y)

rf_cv_model

best_params = rf_cv_model.best_params_
print(best_params)

rf = RandomForestClassifier(
    max_features = best_params['max_features']
    , min_samples_split = best_params['min_samples_split']
    , n_estimators = best_params['n_estimators']
).fit(X, y)

y_pred = rf.predict(X)

from sklearn.metrics import accuracy_score
accuracy_score(y, y_pred)

rf.feature_importances_

feature_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10,7))
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.grid(True)
plt.show()

from sklearn.model_selection import cross_val_score
cross_val_score(rf, X, y, cv=7).mean()

plt.figure(figsize=(10,7))
cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cbar=True)
plt.title('confusion matrix')

from sklearn.metrics import roc_auc_score, roc_curve
rf_roc_auc = roc_auc_score(y, rf.predict(X))
fpr, tpr, thresholds = roc_curve(y, rf.predict_proba(X)[:, 1])
plt.figure(figsize=(10,7))
plt.plot(fpr, tpr, label = 'AUC (area= %0.2f)' % rf_roc_auc)
plt.plot([0,1], [0,1], 'r--')
plt.legend(loc='lower right')
plt.show()

test = test.drop(['Survived'], axis=1)
submission = df_test.PassengerId.copy().to_frame()
prediction = rf.predict(test)

prediction = [int(i) for i in prediction]
submission['Survived'] = prediction

submission.to_csv('submission.csv', index=False)

from google.colab import files
files.download('submission.csv')

