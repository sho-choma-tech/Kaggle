import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 


# 特徴量重要度を棒グラフでプロットする関数 
def plot_feature_importance(df): 
    n_features = len(df)  # 特徴量数(説明変数の個数) 
    df_plot = df.sort_values('importance')  # df_importanceをプロット用に特徴量重要度を昇順ソート 
    f_importance_plot = df_plot['importance'].values  # 特徴量重要度の取得 
    plt.barh(range(n_features), f_importance_plot, align='center') 
    cols_plot = df_plot['feature'].values             # 特徴量の取得 
    plt.yticks(np.arange(n_features), cols_plot)      # x軸,y軸の値の設定
    plt.xlabel('Feature importance')                  # x軸のタイトル
    plt.ylabel('Feature')


# 特徴量重要度の算出 (データフレームで取得)
cols = list(train_df_.drop('delayTime',axis=1).columns) # 特徴量名のリスト(目的変数CRIM以外)
f_importance = np.array(model_lgb.feature_importance()) # 特徴量重要度の算出
f_importance = f_importance / np.sum(f_importance)  # 正規化(必要ない場合はコメントアウト)
df_importance = pd.DataFrame({'feature':cols, 'importance':f_importance})
#df_importance = df_importance.sort_values('importance', ascending=False) # 降順ソート
#display(df_importance)
f_importance


# 特徴量重要度の可視化
plot_feature_importance(df_importance)



# ------------------------------------------------------
# EDA
# ------------------------------------------------------

# データは何日分なのか
d = train_df['date'].value_counts().reset_index()
d.columns = ['date', 'count']
d = d.sort_values('date', ascending=True).reset_index(drop=True)


# 日付ごとのデータの数の分布
d['date'] = list(map(lambda x: str(x), d['date']))

plt.figure(figsize=(20, 5))
plt.bar(d['date'], d['count'])
plt.xticks(rotation=60)
plt.grid()