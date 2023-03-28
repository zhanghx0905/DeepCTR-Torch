# -*- coding:utf-8 -*-
"""
Author:
    Wutong Zhang
Reference:
    [1] Guo H, Tang R, Ye Y, et al. Deepfm: a factorization-machine based neural network for ctr prediction[J]. arXiv preprint arXiv:1703.04247, 2017.(https://arxiv.org/abs/1703.04247)
"""

import torch
import torch.nn as nn

from .basemodel import BaseModel
from ..inputs import combined_dnn_input
from ..layers import DNN, CIN

# 包含三个模块，Linear模块使用原始特征做LR，DNN模块使用转换为embedding后的稀疏特征以及密集特征做MLP，用于隐式提取高阶的特征交互，CIN模块使用同样的embedding显示的提取高阶特征交互
# 三个模块的输出结果相加再经过Sigmoid获得结果
class xDeepFM(BaseModel):
    """Instantiates the xDeepFM architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param cin_layer_size: list,list of positive integer or empty list, the feature maps  in each hidden layer of Compressed Interaction Network
    :param cin_split_half: bool.if set to True, half of the feature maps in each hidden will connect to output unit
    :param cin_activation: activation function used on feature maps
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: L2 regularizer strength applied to deep net
    :param l2_reg_cin: L2 regularizer strength applied to CIN.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.

    """
    # 添加参数use_dense_embedding: bool. 如果设置为True, 在CIN的输入加入扩展到10维的密集特征
    # change dnn_hidden_units=(256, 256) -> (400, 400)
    # change cin_layer_size=(256, 128,) -> (200, 200, 200, )
    # change cin_activation='relu' -> 'linear'
    # change dnn_dropout=0 -> 0.5

    def __init__(self, linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(400, 400),
                 cin_layer_size=(200, 200, 200, ), cin_split_half=True, cin_activation='linear', l2_reg_linear=0.00001,
                 l2_reg_embedding=0.00001, l2_reg_dnn=0, l2_reg_cin=0, init_std=0.0001, seed=1024, dnn_dropout=0.5,
                 dnn_activation='relu', dnn_use_bn=False, task='binary', device='cpu', gpus=None, use_dense_embedding=False):

        super(xDeepFM, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                      l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                      device=device, gpus=gpus)
        self.dnn_hidden_units = dnn_hidden_units
        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0
        # DNN模块, 用于隐式提取高阶特征交互
        # 通过两层的线性层+relu实现
        # 在位维度(bit level)来获得特征交互的结果
        # DNN输入为: 经过embedding table后的稀疏特征与归一化后的密集特征拼接而成
        if self.use_dnn:
            # compute_input_dim计算输入特征维度 = 稀疏特征数 × 稀疏特征转换为的embedding维度 + 密集特征数 × 1
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
            self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)

            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)

        self.cin_layer_size = cin_layer_size
        self.use_cin = len(self.cin_layer_size) > 0 and len(dnn_feature_columns) > 0
        self.use_dense_embedding = use_dense_embedding
        # CIN模块, 用于显示提取高阶的特征交互
        # 每一层用通过上一层的特征交互结果xk-1与原始特征x0交互得到
        # 交互分成两个阶段, 首先计算不同特征之间的Hadamard product，之后再进行卷积操作。
        # 每个卷积核沿着embedding_dim方向滑动, 获得一个长度为embedding_dim的feature map
        # cin_layer_size规定了每一层不同卷积核的数量, 也就是不同feature map的数量, 不同feature map组合起来即为本层的特征交互结果
        # cin_split_half为True时，本层计算的特征交互结果一半用于输出, 一半用于下一层特征交互的计算
        # 最终不同层的特征交互沿着embedding_dim方向求和, 结果拼接起来, 作为CIN的输出
        # CIN模块在向量维度(vector level)获得特征交互的结果
        if self.use_cin:
            # field_num为CIN输入层特征总数, 即x0的特征数
            # 不使用密集特征时, field_num为稀疏特征数（即embedding表的总数）
            # 使用密集特征时, field_num为稀疏特征数+密集特征数（criteo数据集为26+13=39）
            if self.use_dense_embedding:
                field_num = 39
            else:
                field_num = len(self.embedding_dict)
            
            if cin_split_half == True:
                self.featuremap_num = sum(
                    cin_layer_size[:-1]) // 2 + cin_layer_size[-1]
            else:
                self.featuremap_num = sum(cin_layer_size)
            self.cin = CIN(field_num, cin_layer_size,
                           cin_activation, cin_split_half, l2_reg_cin, seed, device=device)
            self.cin_linear = nn.Linear(self.featuremap_num, 1, bias=False).to(device)
            self.add_regularization_weight(filter(lambda x: 'weight' in x[0], self.cin.named_parameters()),
                                           l2=l2_reg_cin)

        # self.final_layer = nn.Linear(3, 1)
        self.to(device)

    def forward(self, X):
        # 将X中稀疏特征与密集特征分开, 稀疏特征通过self.embedding_dict转换为embedding
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,self.embedding_dict)

        # 如果在CIN中使用密集特征, 将1维归一化后的密集特征值重复扩展到与稀疏特征embedding同维度
        if self.use_dense_embedding:       
            dense_embedding_list = []                                                                
            for dense_feature in dense_value_list:
                dense_embedding_list.append(dense_feature.unsqueeze(1).repeat(1, 1, 10))

        # 线性层输出（输入: 原始特征）
        linear_logit = self.linear_model(X)
        # CIN输出
        # （use_dense_embedding = True时, 输入: 1稀疏特征映射得到的10维embedding + 密集特征扩展得到的10维embedding）
        # （use_dense_embedding = False时, 输入: 1稀疏特征映射得到的10维embedding）
        if self.use_cin:
            if self.use_dense_embedding:
                cin_input = torch.cat(sparse_embedding_list + dense_embedding_list, dim=1) # 稀疏特征 + 密集特征拼接         
            else:
                cin_input = torch.cat(sparse_embedding_list, dim=1) # 稀疏特征拼接
            cin_output = self.cin(cin_input)
            cin_logit = self.cin_linear(cin_output)
        # DNN输出（输入: 1稀疏特征映射得到的10维embedding + 1维密集特征值）
        if self.use_dnn:
            dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)  

        if len(self.dnn_hidden_units) == 0 and len(self.cin_layer_size) == 0:  # only linear
            final_logit = linear_logit
        elif len(self.dnn_hidden_units) == 0 and len(self.cin_layer_size) > 0:  # linear + CIN
            final_logit = linear_logit + cin_logit                                                    
        elif len(self.dnn_hidden_units) > 0 and len(self.cin_layer_size) == 0:  # linear +　Deep
            final_logit = linear_logit + dnn_logit
        elif len(self.dnn_hidden_units) > 0 and len(self.cin_layer_size) > 0:  # linear + CIN + Deep
            final_logit = linear_logit + dnn_logit + cin_logit
            # final = torch.cat((linear_logit, cin_logit, dnn_logit), dim=-1)
            # final_logit = self.final_layer(final)
        else:
            raise NotImplementedError

        y_pred = self.out(final_logit)

        return y_pred
