# -*- coding: utf-8 -*-
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import *

import argparse
import os

def seed_everything(seed_value):
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

seed_everything(seed_value=17)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dense_embedding', action='store_true')
    parser.add_argument('--use_sample_data', action='store_true')
    parser.add_argument('--not_use_cin', action='store_true')
    parser.add_argument('--not_use_dnn', action='store_true')
    parser.add_argument('--save_path', type=str, default="ckpt/raw.pth")
    args = parser.parse_args()
    print("use dense embedding :", args.dense_embedding)
    print("use cin: ", not args.not_use_cin)
    print("use dnn: ", not args.not_use_dnn)

    os.makedirs("ckpt", exist_ok=True)

    # 稀疏特征与密集特征名，总共包含26个稀疏特征以及13个密集特征
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    if args.use_sample_data:
        data = pd.read_csv('./examples/criteo_sample.txt')
    else:
        n_sample = 45840617
        n_used_sample = int(n_sample)

        train_file = "/ceph/home/caiyr18/hdd/xdeepfm/criteo/train.txt"
        test_file = "/ceph/home/caiyr18/xdeepfm/criteo/test.txt"

        # 读取数据
        train_data = pd.read_csv(train_file, sep='\t', header=None, names=['label'] + dense_features + sparse_features,
                                nrows=n_used_sample)
        # test_data = pd.read_csv(test_file, sep='\t', header=None, names=dense_features + sparse_features,
        #                         nrows=n_used_sample)

        print(len(train_data))

        data = train_data

    # print(data)
    # 将nan的特征补全，稀疏特征补-1，密集特征补0
    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    # 对原始数据特征预处理
    # 稀疏特征编码为0～N-1之间的整数，其中N为该稀疏特征的所有可能取值
    # 密集特征归一化到0-1
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name
    # 构建特征列
    # 对稀疏特征:记录每个稀疏特征的名字、可能的取值数量（vocabulary_size）和对应embedding的维度
    # 对密集特征:记录每个密集特征的名字，维度（dimension）默认为1
    # change 4->10（稀疏特征对应embedding的维度）
    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=10)
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                              for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
    # 8:2分离训练集以及测试集
    train, test = train_test_split(data, test_size=0.2, random_state=2022)
    # train, test = data[:n_used_sample], data[n_used_sample:]
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate
    # 训练所用的cpu或gpu
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'
    # 定义模型
    dnn_hidden_units= (400, 400) if not args.not_use_dnn else ()
    cin_layer_size= (200, 200, 200, ) if not args.not_use_cin else ()
    print("dnn_hidden_units: ", dnn_hidden_units)
    print("cin_layer_size: ", cin_layer_size)
    model = xDeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                    dnn_hidden_units=dnn_hidden_units, cin_layer_size=cin_layer_size,
                   task='binary',
                   l2_reg_embedding=1e-5, device=device, use_dense_embedding=args.dense_embedding)
    # 定义优化器、损失函数以及评价指标
    # change adagrad -> adam
    model.compile("adam", "binary_crossentropy",
                  metrics=["binary_crossentropy", "auc"], )
    # 训练函数 
    # 包含分割训练集以及验证集
    # 训练epochs轮，每轮训练首先在训练集上以batch_size大小的数据进行迭代，包括模型前向传播，计算损失，梯度反向传播并更新模型参数
    # 每个epoch中模型在训练集上训练一遍后在验证集上测试binary_crossentropy以及auc
    model_path = args.save_path
    print("save model in ", model_path)
    history = model.fit(train_model_input, train[target].values, batch_size=1024, epochs=10, verbose=2,
                        validation_split=0.2, model_path=model_path)
    model.load_state_dict(torch.load(model_path))
    
    pred_ans = model.predict(test_model_input, 256)
    print("")
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
