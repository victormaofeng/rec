import math
from typing import List

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class DCN_V2_Net(nn.Layer):
    """
    两种模式：
    Model Structaul: Stacked or Parallel

    网络 的 输入维度 == 输出维度
    """

    def __init__(self, layer_sizes: List[int], cross_num: int, input_size: int,
                 is_stacked: bool, use_low_rank_mixture: bool,
                 low_rank: int, num_experts: int):

        super(DCN_V2_Net, self).__init__()

        self.layer_sizes = layer_sizes
        self.cross_num = cross_num
        self.is_stacked = is_stacked
        self.use_low_rank_mixture = use_low_rank_mixture
        self.low_rank = low_rank
        self.num_experts = num_experts
        self.input_size = input_size

        self.init_value_ = 0.1

        use_sparse = True
        if paddle.is_compiled_with_custom_device('npu'):
            use_sparse = False

        self.deep_cross_layer = DeepCrossLayer(
            input_size, cross_num,
            use_low_rank_mixture, low_rank, num_experts)

        self.dnn_layer = DNNLayer(
            input_size,
            layer_sizes,
            dropout_rate=0.5)

        # fc_sizes: [768, 768]

        if self.is_stacked:
            self.fc = paddle.nn.Linear(
                in_features=input_size,
                out_features=input_size,
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(input_size))))

        else:
            self.fc = paddle.nn.Linear(
                in_features=input_size * 2,
                out_features=input_size,
                weight_attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(input_size * 2))))

    def forward(self, x):

        # Model Structaul: Stacked or Parallel
        if self.is_stacked:
            # CrossNetLayer
            cross_out = self.deep_cross_layer(x)
            # MLPLayer
            dnn_output = self.dnn_layer(cross_out)
            # print('----dnn_output shape----',dnn_output.shape)
            logit = self.fc(dnn_output)
            predict = F.sigmoid(logit)

        else:
            # CrossNetLayer
            cross_out = self.deep_cross_layer(x)
            # MLPLayer
            dnn_output = self.dnn_layer(x)
            last_out = paddle.concat([dnn_output, cross_out], axis=-1)
            # print('----last_out_output shape----',last_out.shape)=
            logit = self.fc(last_out)
            predict = F.sigmoid(logit)
        return predict


class DNNLayer(paddle.nn.Layer):
    def __init__(self, input_size: int,
                 layer_sizes,
                 dropout_rate=0.5):
        super(DNNLayer, self).__init__()

        self.layer_sizes = layer_sizes

        self.input_size = input_size

        self.drop_out = paddle.nn.Dropout(p=dropout_rate)

        sizes = [input_size] + self.layer_sizes + [input_size]

        acts = ["relu" for _ in range(len(sizes) - 1)] + [None]

        self._mlp_layers = []
        for i in range(len(sizes) - 1):
            linear = paddle.nn.Linear(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                weight_attr=paddle.ParamAttr(
                    regularizer=paddle.regularizer.L2Decay(1e-7),
                    initializer=paddle.nn.initializer.Normal(
                        std=1.0 / math.sqrt(sizes[i]))))
            self.add_sublayer('linear_%d' % i, linear)
            self._mlp_layers.append(linear)
            if acts[i] == 'relu':
                act = paddle.nn.ReLU()
                self.add_sublayer('act_%d' % i, act)
                self._mlp_layers.append(act)

    def forward(self, feat_embeddings):
        # y_dnn = paddle.reshape(feat_embeddings,[feat_embeddings.shape[0], -1])
        y_dnn = feat_embeddings
        for n_layer in self._mlp_layers:
            y_dnn = n_layer(y_dnn)
            y_dnn = self.drop_out(y_dnn)
        return y_dnn


class DeepCrossLayer(nn.Layer):
    def __init__(self, input_size: int,
                 cross_num, use_low_rank_mixture, low_rank, num_experts):
        super(DeepCrossLayer, self).__init__()

        self.use_low_rank_mixture = use_low_rank_mixture

        self.input_size = input_size

        self.num_experts = num_experts
        self.low_rank = low_rank
        self.cross_num = cross_num

        if self.use_low_rank_mixture:
            self.crossNet = CrossNetMix(
                self.input_size,
                layer_num=self.cross_num,
                low_rank=self.low_rank,
                num_experts=self.num_experts)
        else:
            self.crossNet = CrossNetV2(self.input_size, self.cross_num)

    def forward(self, feat_embeddings):
        outputs = self.crossNet(feat_embeddings)
        return outputs


class CrossNetV2(nn.Layer):
    def __init__(self, input_dim: int, num_layers):
        super(CrossNetV2, self).__init__()

        self.num_layers = num_layers
        self.cross_layers = nn.LayerList(
            nn.Linear(input_dim, input_dim) for _ in range(self.num_layers))

    def forward(self, X_0):
        X_i = X_0  # b x dim
        for i in range(self.num_layers):
            X_i = X_i + X_0 * self.cross_layers[i](X_i)
        return X_i


class CrossNetMix(nn.Layer):
    """
    CrossNetMix improves CrossNet by:
        1. add MOE to learn feature interactions in different subspaces
        2. add nonlinear transformations in low-dimensional space
    将高秩向量转为低秩向量，减少模型参数
    """

    def __init__(self, in_features, layer_num=2, low_rank=32, num_experts=4):
        super(CrossNetMix, self).__init__()
        self.layer_num = layer_num
        self.num_experts = num_experts

        # U: (in_features, low_rank)
        self.U_list = paddle.nn.ParameterList([
            paddle.create_parameter(
                shape=[num_experts, in_features, low_rank],
                dtype='float32',
                default_initializer=paddle.nn.initializer.XavierNormal())
            for i in range(self.layer_num)
        ])

        # V: (in_features, low_rank)
        self.V_list = paddle.nn.ParameterList([
            paddle.create_parameter(
                shape=[num_experts, in_features, low_rank],
                dtype='float32',
                default_initializer=paddle.nn.initializer.XavierNormal())
            for i in range(self.layer_num)
        ])

        # C: (low_rank, low_rank)
        self.C_list = paddle.nn.ParameterList([
            paddle.create_parameter(
                shape=[num_experts, low_rank, low_rank],
                dtype='float32',
                default_initializer=paddle.nn.initializer.XavierNormal())
            for i in range(self.layer_num)
        ])

        self.gating = nn.LayerList(
            [nn.Linear(in_features, 1) for i in range(self.num_experts)])

        self.bias = paddle.nn.ParameterList([
            paddle.create_parameter(
                shape=[in_features, 1],
                dtype='float32',
                default_initializer=paddle.nn.initializer.Constant(value=0.0))
            for i in range(self.layer_num)
        ])

    def forward(self, inputs):
        x_0 = inputs.unsqueeze(2)  # (bs, in_features, 1)
        x_l = x_0
        for i in range(self.layer_num):
            output_of_experts = []
            gating_score_of_experts = []
            for expert_id in range(self.num_experts):
                # (1) G(x_l)
                # compute the gating score by x_l
                gating_score_of_experts.append(self.gating[expert_id](
                    x_l.squeeze(2)))

                # (2) E(x_l)
                # project the input x_l to $\mathbb{R}^{r}$
                v_x = paddle.matmul(self.V_list[i][expert_id].t(),
                                    x_l)  # (bs, low_rank, 1)

                # nonlinear activation in low rank space
                v_x = paddle.tanh(v_x)
                v_x = paddle.matmul(self.C_list[i][expert_id], v_x)
                v_x = paddle.tanh(v_x)

                # project back to $\mathbb{R}^{d}$
                uv_x = paddle.matmul(self.U_list[i][expert_id],
                                     v_x)  # (bs, in_features, 1)

                dot_ = uv_x + self.bias[i]
                dot_ = x_0 * dot_  # Hadamard-product

                output_of_experts.append(dot_.squeeze(2))

            # (3) mixture of low-rank experts
            output_of_experts = paddle.stack(
                output_of_experts, axis=2)  # (bs, in_features, num_experts)
            gating_score_of_experts = paddle.stack(
                gating_score_of_experts, axis=1)  # (bs, num_experts, 1)
            moe_out = paddle.matmul(
                output_of_experts, F.softmax(
                    gating_score_of_experts, axis=1))
            x_l = moe_out + x_l  # (bs, in_features, 1)

        x_l = x_l.squeeze()  # (bs, in_features)
        return x_l


if __name__ == '__main__':
    net = DCN_V2_Net(layer_sizes=[768, 768], cross_num=2,
                     input_size=1024, is_stacked=True,
                     use_low_rank_mixture=False,
                     low_rank=256, num_experts=4)
    x = paddle.randn([8, 1024])
    p = net(x)
    print("")
