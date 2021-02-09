# -*- coding: utf-8 -*-
# how to generate preprocessed csv and pkl files
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import *

data = pd.read_csv('/home/usr1/zd2922/avazu_dir/train')
#header_names = data.columns.tolist()
header_names = ['id', 'click', 'hour', 'C1', 'banner_pos', 'site_id', 'site_domain',
       'site_category', 'app_id', 'app_domain', 'app_category', 'device_id',
       'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14',
       'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
sparse_features = header_names[3:]
dense_features = ['hour']
data[sparse_features] = data[sparse_features].fillna('-1', )
data[dense_features] = data[dense_features].fillna(0, )
target = ['click']
# 1.Label Encoding for sparse features,and do simple Transformation for dense features
for feat in sparse_features:
    print('Label Encoding: ',feat)
    print('dimension: ',data[feat].nunique())
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])

mms = MinMaxScaler(feature_range=(0, 1))
data[dense_features] = mms.fit_transform(data[dense_features])

type_dict = {name:'float32' for name in dense_features}
type_dict.update({name: 'int32' for name in sparse_features})
type_dict.update({name:"int32" for name in target})
for col,dt in type_dict.items():
   print(col,dt)
   data[col] = data[col].astype(dt)
data = data.drop('id',axis=1)
data.to_pickle('preprocessed_avazu.pkl')
