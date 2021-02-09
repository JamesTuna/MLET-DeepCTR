# -*- coding: utf-8 -*-
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import *
import pickle
import argparse
#from apex import amp
import torch
# parser
parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, default=256, help='batch_size')
parser.add_argument('--inner_dim', type=int, default=12, help='inner dimension of embedding factorization')
parser.add_argument('--embd_dim', type=int, default=4, help='embedding dimension')
parser.add_argument('--optimizer',type=str,default='adagrad',help='optimizer')
parser.add_argument('--epoch',type=int,default=1,help='number of epochs to train')
parser.add_argument('--lr',type=float,default=0.01,help='learning rate')
parser.add_argument('--gpu_number',type=int,default=0,help='which gpu to use')
#parser.add_argument('--amp',type=int,default=1,help='automatic mixed precision')
args = parser.parse_args()

INNER_DIM = args.inner_dim
if INNER_DIM <= 0:
    INNER_DIM = None
BATCH = args.batch
OUTER_DIM = args.embd_dim
#data = pd.read_csv('../../preprocessed/criteo_train.csv')
data = pickle.load(open('../preprocessed/preprocessed_avazu.pkl','rb'))
header_names = ['id', 'click', 'hour', 'C1', 'banner_pos', 'site_id', 'site_domain',
       'site_category', 'app_id', 'app_domain', 'app_category', 'device_id',
       'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14',
       'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
sparse_features = header_names[3:]
dense_features = ['hour']
target = ['click']
# 2.count #unique features for each sparse field,and record dense feature field name
fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(),embedding_dim=OUTER_DIM) for feat in sparse_features] + [DenseFeat(feat, 1, ) for feat in dense_features]
dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
# 3.generate input data for model
train, test = train_test_split(data, test_size=0.1,random_state=42)
train_model_input = {name: train[name] for name in feature_names}
test_model_input = {name: test[name] for name in feature_names}
train_model_labels = train[target].values
test_model_labels = test[target].values
# memory optimization
import gc
del data
data = None
gc.collect()
# 4.Define Model,train,predict and evaluate
device = 'cpu'
use_cuda = True
if use_cuda and torch.cuda.is_available():
    print('cuda ready...')
    device = 'cuda:%s'%args.gpu_number

print('Training xDeepFM model with inner dim: %s...'%(INNER_DIM))
model = xDeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,task='binary',l2_reg_embedding=1e-5, device=device, embd_inner_dim = INNER_DIM)

# print model parameter informationmodel.state_dict()
total_para = 0
for name,para in model.state_dict().items():
    print(name,'\t',para.dtype,'\t',para.size())
    total_para += para.numel()

print('model parameters(fp32) in GB: ',total_para*4/1024**3)


model.compile("adagrad", "binary_crossentropy", metrics=["binary_crossentropy", "auc"], ) # default
# create customized optimizer
del model.optim
if args.optimizer == "sgd":
    model.optim = torch.optim.SGD(model.parameters(), lr=args.lr)
elif args.optimizer == "adam":
    model.optim = torch.optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer == "adagrad":
    model.optim = torch.optim.Adagrad(model.parameters(),lr=args.lr)
elif args.optimizer == "rmsprop":
    model.optim = torch.optim.RMSprop(model.parameters())
else:
    raise NotImplementedError

#if args.amp > 0:
#    print("enable amp")
#    model, model.optim = amp.initialize(model, model.optim, opt_level="O1")
for ep in range(args.epoch):
    print("Trainer: %s/%s epoch training starts..."%(ep+1,args.epoch))
    model.fit(train_model_input, train_model_labels,batch_size=BATCH, epochs=1, validation_split=0.0, verbose=2, amp_enable= False)
    pred_ans = model.predict(test_model_input, BATCH)
    print("test LogLoss", round(log_loss(test_model_labels, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test_model_labels, pred_ans), 4))
    print("")
