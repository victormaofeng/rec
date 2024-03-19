# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pdb

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math
import numpy as np
import os
import sys

from models.demo.system.base.dcn_v2 import DCN_V2_Net


class DssmNet(nn.Layer):
    """
    双塔模型

    内部采用 DCN 特征交叉网络
    """

    def __init__(self, sparse_feature_number, sparse_feature_dim, fc_sizes):
        super(DssmNet, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.fc_sizes = fc_sizes

        self.embedding = paddle.nn.Embedding(
            self.sparse_feature_number,
            self.sparse_feature_dim,
            padding_idx=0,
            sparse=True,
            weight_attr=paddle.ParamAttr(
                name="SparseFeatFactors",
                initializer=paddle.nn.initializer.Uniform()))

        self.user_net = DCN_V2_Net(layer_sizes=[512, 256, 128], cross_num=2,
                                   input_size=36, is_stacked=True,
                                   use_low_rank_mixture=True,
                                   low_rank=32, num_experts=4)

        self.movie_pre_layer = paddle.nn.Linear(
            in_features=27,
            out_features=36,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Normal(
                    std=1.0 / math.sqrt(27))))

        self.movie_net = DCN_V2_Net(layer_sizes=[512, 256, 128], cross_num=2,
                                    input_size=36, is_stacked=True,
                                    use_low_rank_mixture=True,
                                    low_rank=32, num_experts=4)

    def forward(self, batch_size, user_sparse_inputs, mov_sparse_inputs,
                label_input):

        user_sparse_embed_seq = []
        for s_input in user_sparse_inputs:
            emb = self.embedding(s_input)
            emb = paddle.reshape(emb, shape=[-1, self.sparse_feature_dim])
            user_sparse_embed_seq.append(emb)

        mov_sparse_embed_seq = []
        for s_input in mov_sparse_inputs:
            # s_input = paddle.reshape(s_input, shape=[batch_size, -1])
            emb = self.embedding(s_input)
            emb = paddle.sum(emb, axis=1)
            emb = paddle.reshape(emb, shape=[-1, self.sparse_feature_dim])
            mov_sparse_embed_seq.append(emb)

        user_features = paddle.concat(user_sparse_embed_seq, axis=1)
        mov_features = paddle.concat(mov_sparse_embed_seq, axis=1)

        # pdb.set_trace()

        user_features = self.user_net(user_features)

        mov_features = self.movie_pre_layer(mov_features)
        mov_features = self.movie_net(mov_features)

        sim = F.cosine_similarity(
            user_features, mov_features, axis=1).reshape([-1, 1])

        # 相似度取值范围0-1，评分为1-5，因此需要将预测的值乘以5
        predict = paddle.scale(sim, scale=5)

        return predict
