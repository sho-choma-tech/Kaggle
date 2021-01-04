import pandas as pd
import numpy as np
import datetime 
from pathlib import Path

import warnings 
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import xgboost as xgb 
import lightgbm as lgb 

DATA_DIR = Path('./data/input')

train = pd.read_csv(DATA_DIR / 'train.csv')
test = pd.read_csv(DATA_DIR / 'test.csv')
info = pd.read_csv(DATA_DIR / 'info.csv')
network = pd.read_csv(DATA_DIR / 'network.csv')

train_df = train.copy()
test_df = test.copy()
info_df = info.copy()
network_df = network.copy()

train_df['mark'] = 'train'
test_df['mark'] = 'test'


# ------------------------------------------------
# データ前処理
# ------------------------------------------------

def lgb_process(train_df, test_df):

    # 時間関連
    # 年度と月は一律だからあまり関係ない → 日付ごとで取得
    # date_hourから日付
    all_data = pd.concat([train_df, test_df], axis=0)
    all_data['date_day'] = abs(all_data['date']) % 100


    # planArrivalのminutesを除外，hoursを取得
    all_data['planArrival'] = pd.to_datetime(all_data['planArrival'], format='%H:%M')
    all_data['plan_hour'] = all_data['planArrival'].dt.hour


    # 訓練データとテストデータの分割
    train_df_ = all_data[all_data['mark'] == 'train']
    train_df_ = train_df_.drop(columns=['date', 'planArrival', 'target'])

    test_df_ = all_data[all_data['mark'] == 'test']
    test_df_ = test_df_.drop(columns=['date', 'planArrival'])
    return train_df_, test_df_


# 隣接駅の取得
def station_process(train_df, test_df, network_df):
    #訓練データの停車駅を取得
    all_stopStation = list(pd.unique(train_df['stopStation']))
    to_dct = {}

    for s in all_stopStation:
        to_lst = []
        for i in range(len(network_df)):
            if network_df['station1'].iloc[i] == s:
                to_lst.append(network_df['station2'].iloc[i])
            if network_df['station2'].iloc[i] == s:
                to_lst.append(network_df['station1'].iloc[i])
        to_dct[s] = to_lst

    # stopStationに対する隣接駅を表すデータフレーム
    connect_station_col = [
        'stopStation'
        ,'conn1'
        ,'conn2'
        ,'conn3'
        ,'conn4'
        ,'conn5'
        ,'numOfConn'
    ]
    connect_station_df = pd.DataFrame(None, columns=connect_station_col)

    connect_station_df['stopStation'] = all_stopStation
    cols = ['conn1', 'conn2', 'conn3', 'conn4', 'conn5']
    for i in range(len(connect_station_df)):
        s = connect_station_df['stopStation'].iloc[i]
        idx = len(to_dct[s])
        for j in range(idx):
            connect_station_df[cols[j]].iloc[i] = to_dct[s][j]
        connect_station_df['numOfConn'].iloc[i] = len(to_dct[s])

    for col in cols:
        connect_station_df[col] = connect_station_df[col].fillna('0')

    train_connect_df = train_df.merge(connect_station_df)
    test_connect_df = test_df.merge(connect_station_df)

    train_connect_df['numOfConn'] = train_connect_df['numOfConn'].astype(int)
    test_connect_df['numOfConn'] = test_connect_df['numOfConn'].astype(int)

    return train_connect_df, test_connect_df



#ラベルエンコーディング
def label_preprocess(train_df, test_df):

    train_x = train_df.drop(['delayTime'], axis=1)
    train_y = train_df['delayTime']
    test_x  = test_df.drop(['delayTime', 'target'], axis=1)
    test_y  = test_df['delayTime']

    cat_cols = [
        'lineName', 'trainNo', 'stopStation'
        , 'lineName_ChuoSpohbu', 'lineName_Keihin', 'lineName_yamanote', 'lineName_ChuoRapid'
        ,'conn1', 'conn2', 'conn3', 'conn4', 'conn5'
    ]

    for c in cat_cols:
        le = LabelEncoder()
        le.fit(train_x[c])
        train_x[c] = le.transform(train_x[c])
        
        le.fit(test_x[c])
        test_x[c] = le.transform(test_x[c])

    return train_x, train_y, test_x, test_y



# 路線情報追加
def line_process(train_df, test_df):

    all_data = pd.concat([train_df, test_df], axis=0)


    #中央総武線データ
    Chuo_Sohbu_dict = {}
    Chuo_Sohbu_list = [
        'Rlfq', 'coZB', 'LMww', 'VNyR', 'jhlV', 'efzW', 'PcxI', 'ejfb'
        ,'RDLf', 'cRgf', 'SvNu', 'GLXs', 'iHon', 'OJng', 'HOLt', 'qzrJ'
        ,'NqOG', 'KsBm', 'QCTc', 'fpRv', 'zUdG', 'APgF', 'RUhy', 'tPfo'
        ,'jPbe', 'stJE', 'ufGe', 'tncu', 'wbkB', 'aCxM', 'AFTQ', 'CTnl'
        ,'mxQg', 'PsTo', 'jebQ', 'dJlm', 'Femc', 'TlTg', 'daXW', 'trdb'
    ]
    A_val = 'A'
    Chuo_Sohbu_dict = { i : A_val for i in Chuo_Sohbu_list }


    #京浜東北線データ
    keihin_dict = {}
    keihin_list = [
        'FPFv', 'mjPK', 'uQdT', 'zNdV', 'KReX', 'aVeu', 'oNkU', 'vcLy'
        ,'cvCM', 'EoNR', 'qdzg', 'lCym', 'DTSj', 'hwjA', 'KpNq', 'ZcpD'
        ,'QkqQ', 'TaOj', 'yuIQ', 'TlTg', 'KsBm', 'ReJL', 'yUYG', 'CbCQ'
        ,'aMqH', 'bgbz', 'LKUe', 'LglK', 'cbJh', 'daBI', 'MQfL', 'jdBM'
        ,'xKSN', 'qCwi', 'sLAH', 'ugLy', 'GsES', 'ntaA', 'Zlsb', 'qzWm'
        ,'UUlH', 'lUUx', 'rGbD', 'OpgJ', 'PYtN'
    ]
    B_val = 'B'
    keihin_dict = { i : B_val for i in keihin_list }


    #山手線データ
    yamanote_dict = {}
    yamanote_list = [
        'hwjA', 'KpNq', 'ZcpD', 'QkqQ', 'TaOj', 'yuIQ', 'TlTg', 'KsBm'
        ,'ReJL', 'yUYG', 'CbCQ', 'aMqH', 'bgbz', 'LKUe', 'fmwC', 'NLJh'
        ,'IXNJ', 'mKyE', 'tNrH', 'daSA', 'Gftx', 'RDLf', 'cRgf', 'djPS'
        ,'fRXM', 'vzmD', 'fXMY', 'SAOS', 'thmK', 'PYtN'
    ]  
    C_val = 'C'
    yamanote_dict = { i : C_val for i in yamanote_list}


    #中央快速線データ
    Chuo_Rapid_dict = {}
    Chuo_Rapid_list = [
        'uYlv', 'AVjc', 'BCRD', 'mkGW', 'UMoa', 'fZfY', 'Hzeg', 'GgOD'
        ,'tCey', 'trdb', 'daXW', 'Rlfq', 'coZB', 'LMww', 'VNyR', 'jhlV'
        ,'efzW', 'RDLf', 'iHon', 'NqOG', 'TlTg', 'wwYD', 'PcxI', 'ejfb'
        ,'cRgf', 'SvNu', 'GLXs', 'OJng', 'HOLt', 'qzrJ', 'KsBm', 'QCTc'
        ,'fpRv', 'zUdG', 'APgF', 'RUhy', 'tPfo', 'jPbe', 'stJE', 'ufGe'
        ,'tncu', 'wbkB', 'aCxM', 'AFTQ', 'CTnl', 'mxQg', 'PsTo', 'jebQ'
        ,'dJlm', 'Femc', 'GxuL'
    ]
    D_val = 'D'
    Chuo_Rapid_dict = { i : D_val for i in Chuo_Rapid_list}


    #特徴量生成
    all_data['lineName_ChuoSpohbu'] = all_data['stopStation'].map(Chuo_Sohbu_dict)
    all_data['lineName_ChuoSpohbu'] = all_data['lineName_ChuoSpohbu'].fillna('0')

    all_data['lineName_Keihin'] = all_data['stopStation'].map(keihin_dict)
    all_data['lineName_Keihin'] = all_data['lineName_Keihin'].fillna('0')

    all_data['lineName_yamanote'] = all_data['stopStation'].map(yamanote_dict)
    all_data['lineName_yamanote'] = all_data['lineName_yamanote'].fillna('0')

    all_data['lineName_ChuoRapid'] = all_data['stopStation'].map(Chuo_Rapid_dict)
    all_data['lineName_ChuoRapid'] = all_data['lineName_ChuoRapid'].fillna('0')



    # 訓練データとテストデータの分割
    train_df_ = all_data[all_data['mark'] == 'train']
    train_df_ = train_df_.drop(columns=['mark', 'target'])

    test_df_ = all_data[all_data['mark'] == 'test']
    test_df_ = test_df_.drop(columns=['mark'])
    return train_df_, test_df_
