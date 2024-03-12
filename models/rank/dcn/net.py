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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math
import pdb


class DeepCroLayer(nn.Layer):
    """
    dcn cnn 和 dnn 并行连接

    数据集介绍：
    criteo数据集用于广告点击率预估任务（标签：0/1）；其中包含13个dense特征和26个sparse特征；
    数据格式如下：第一列为label, 之后分别是13个dense特征(integer feature)，
    26个sparse特征(categorical feature)；每列之间使用tab进行分隔

    """
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 dense_feature_dim, sparse_num_field, layer_sizes, cross_num,
                 clip_by_norm, l2_reg_cross, is_sparse):
        super(DeepCroLayer, self).__init__()
        #   sparse_feature_number: 1000001
        #   sparse_feature_dim: 9
        #   dense_input_dim: 13
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.sparse_num_field = sparse_num_field
        self.layer_sizes = layer_sizes
        self.cross_num = cross_num
        self.clip_by_norm = clip_by_norm
        self.l2_reg_cross = l2_reg_cross
        self.is_sparse = is_sparse

        self.init_value_ = 0.1

        use_sparse = True
        if paddle.is_compiled_with_custom_device('npu'):
            use_sparse = False

        # sparse coding
        self.embedding = paddle.nn.Embedding(
            self.sparse_feature_number,
            self.sparse_feature_dim,
            sparse=use_sparse,
            padding_idx=0,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.TruncatedNormal(
                    mean=0.0,
                    std=self.init_value_ /
                    math.sqrt(float(self.sparse_feature_dim)))))

        # w
        self.layer_w = paddle.create_parameter(
            shape=[
                self.dense_feature_dim + self.sparse_num_field *
                self.sparse_feature_dim
            ],
            dtype='float32',
            default_initializer=paddle.nn.initializer.TruncatedNormal(
                mean=0.0,
                std=self.init_value_ /
                math.sqrt(float(self.sparse_feature_dim))))

        # b
        self.layer_b = paddle.create_parameter(
            shape=[
                self.dense_feature_dim + self.sparse_num_field *
                self.sparse_feature_dim
            ],
            dtype='float32',
            default_initializer=paddle.nn.initializer.TruncatedNormal(
                mean=0.0,
                std=self.init_value_ /
                math.sqrt(float(self.sparse_feature_dim))))

        # DNN 深度网络 deep NN
        self.num_field = self.dense_feature_dim + self.sparse_num_field * self.sparse_feature_dim
        sizes = [self.num_field] + self.layer_sizes
        acts = ["relu" for _ in range(len(self.layer_sizes))] + [None]
        self._mlp_layers = []
        # fc_sizes: [512, 256, 128]  # , 32]
        for i in range(len(self.layer_sizes)):
            linear = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(sizes[i]))))
            self.add_sublayer('linear_%d' % i, linear)
            self._mlp_layers.append(linear)
            if acts[i] == 'relu':
                act = paddle.nn.ReLU()
                self.add_sublayer('act_%d' % i, act)
                self._mlp_layers.append(act)

        # 最后的全连接层
        self.fc = paddle.nn.Linear(
            in_features=self.layer_sizes[-1] + self.sparse_num_field *
            self.sparse_feature_dim + self.dense_feature_dim,
            out_features=1,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Normal(
                    std=1.0 /
                    math.sqrt(self.layer_sizes[-1] + self.sparse_num_field +
                              self.dense_feature_dim))))

    def _create_embedding_input(self, sparse_inputs, dense_inputs):
        """

        """
        # 合并离散特征 sparse_inputs_concat.shape = [8,26]
        sparse_inputs_concat = paddle.concat(sparse_inputs, axis=1)
        # 对离散特征做embedding  sparse_embeddings.shape=[8,26,9]
        sparse_embeddings = self.embedding(sparse_inputs_concat)

        sparse_embeddings_re = paddle.reshape(
            sparse_embeddings,
            shape=[-1, self.sparse_num_field * self.sparse_feature_dim])
        # 拼接离散特征和连续特征
        feat_embeddings = paddle.concat([sparse_embeddings_re, dense_inputs],
                                        1)

        pdb.set_trace()
        return feat_embeddings

    # 交叉层
    def _cross_layer(self, input_0, input_x):
        input_w = paddle.multiply(input_x, self.layer_w)
        input_w1 = paddle.sum(input_w, axis=1, keepdim=True)

        input_ww = paddle.multiply(input_0, input_w1)

        input_layer_0 = paddle.add(input_ww, self.layer_b)
        input_layer = paddle.add(input_layer_0, input_x)

        return input_layer, input_w

    # 交叉网络
    def _cross_net(self, input, num_corss_layers):
        x = x0 = input
        l2_reg_cross_list = []
        for i in range(num_corss_layers):
            x, w = self._cross_layer(x0, x)
            l2_reg_cross_list.append(self._l2_loss(w))
        l2_reg_cross_loss = paddle.add_n(l2_reg_cross_list)
        return x, l2_reg_cross_loss

    def _l2_loss(self, w):
        return paddle.sum(paddle.square(w))

    def forward(self, sparse_inputs, dense_inputs):

        pdb.set_trace()

        feat_embeddings = self._create_embedding_input(sparse_inputs,
                                                       dense_inputs)
        cross_out, l2_reg_cross_loss = self._cross_net(feat_embeddings,
                                                       self.cross_num)

        dnn_feat = feat_embeddings

        for n_layer in self._mlp_layers:
            dnn_feat = n_layer(dnn_feat)

        last_out = paddle.concat([dnn_feat, cross_out], axis=-1)

        logit = self.fc(last_out)

        predict = F.sigmoid(logit)

        return predict, l2_reg_cross_loss
