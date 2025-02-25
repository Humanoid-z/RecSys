
# from deepctr_torch.layers import DNN
from .PLBaseModel import PLBaseModel
from deepctr_torch.inputs import SparseFeat, DenseFeat, VarLenSparseFeat
from deepctr_torch.inputs import get_varlen_pooling_list, varlen_embedding_lookup
# from ..utils import combined_dnn_input
import torch
import torch.nn as nn
import torch.nn.functional as F


def combined_dnn_input(sparse_embedding_list, dense_value_list=[]):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = torch.flatten(
            torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
        dense_dnn_input = torch.flatten(
            torch.cat(dense_value_list, dim=-1), start_dim=1)
        return concat_fun([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return torch.flatten(torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
    elif len(dense_value_list) > 0:
        return torch.flatten(torch.cat(dense_value_list, dim=-1), start_dim=1)
    else:
        raise NotImplementedError


def activation_layer(act_name, hidden_size=None, dice_dim=2):
    """Construct activation layers

    Args:
        act_name: str or nn.Module, name of activation function
        hidden_size: int, used for Dice activation
        dice_dim: int, used for Dice activation
    Return:
        act_layer: activation layer
    """
    if isinstance(act_name, str):
        if act_name.lower() == 'sigmoid':
            act_layer = nn.Sigmoid()
        elif act_name.lower() == 'linear':
            act_layer = Identity()
        elif act_name.lower() == 'relu':
            act_layer = nn.ReLU(inplace=True)
        elif act_name.lower() == 'dice':
            assert dice_dim
            act_layer = Dice(hidden_size, dice_dim)
        elif act_name.lower() == 'prelu':
            act_layer = nn.PReLU()
    elif issubclass(act_name, nn.Module):
        act_layer = act_name()
    else:
        raise NotImplementedError

    return act_layer


class DNN(nn.Module):

    def __init__(self, inputs_dim, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False,
                 init_std=0.0001, dice_dim=3, seed=1024):
        super(DNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        hidden_units = [inputs_dim] + list(hidden_units)

        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        if self.use_bn:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        self.activation_layers = nn.ModuleList(
            [activation_layer(activation, hidden_units[i + 1], dice_dim) for i in range(len(hidden_units) - 1)])

        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

    def forward(self, inputs):
        deep_input = inputs

        for i in range(len(self.linears)):

            fc = self.linears[i](deep_input)

            if self.use_bn:
                fc = self.bn[i](fc)

            fc = self.activation_layers[i](fc)

            fc = self.dropout(fc)
            deep_input = fc
        return deep_input


# youtube dnn 的结构没有什么改变
class YouTubeDNN(PLBaseModel):
    def __init__(self, user_feature_columns, item_feature_columns, dnn_hidden_units=[64, 32],
                 dnn_activation='relu', dnn_use_bn=False,
                 init_std=0.002,
                 l2_reg_dnn=0, l2_reg_embedding=1e-6,
                 dnn_dropout=0, activation='relu', seed=1024,device='cuda', **kwargs):
        super(YouTubeDNN, self).__init__(user_feature_columns, item_feature_columns,
                                         l2_reg_linear=1e-5, l2_reg_embedding=1e-5,
                                         init_std=0.0001, seed=1024,device=device, **kwargs)

        self.user_feature_columns = user_feature_columns
        self.item_feature_columns = item_feature_columns
        self.user_dnn = DNN(self.compute_input_dim(user_feature_columns), dnn_hidden_units,
                            activation=dnn_activation, init_std=init_std)
        self.item_dnn = DNN(self.compute_input_dim(item_feature_columns), dnn_hidden_units,
                            activation=dnn_activation, init_std=init_std)
        self.to(device)

    def forward(self, X):

        batch_size = X.size(0)
        user_embedding = self.user_tower(X)
        item_embedding = self.item_tower(X)

        if self.mode == "user_representation":
            return user_embedding
        if self.mode == "item_representation":
            return item_embedding

        score = F.cosine_similarity(user_embedding, item_embedding, dim=-1)
        score = score.view(batch_size, -1)
        score = torch.clamp(score, max=1)
        return score

    def item_tower(self, X):
        if self.mode == "user_representation":
            return None
        X = X.to(self.device)
        item_embedding_list, item_content_embedding_list = self.input_from_item_feature_columns(X,
                                                                                                self.item_feature_columns,
                                                                                                self.embedding_dict)
        item_embedding = item_embedding_list[0]  # (batch, movie_list_len, feat_dim)
        if item_content_embedding_list:
            item_content_embedding_list = [ts.unsqueeze(1).type(item_embedding.type()) for ts in
                                           item_content_embedding_list]
            dnn_input = torch.cat([item_embedding, *item_content_embedding_list], dim=-1)
        else:
            dnn_input = item_embedding
        item_embedding = self.item_dnn(dnn_input.to(self.device))
        return item_embedding

    def user_tower(self, X):
        X = X.to(self.device)
        if self.mode == "item_representation":
            return None
        # sample softmax 可以通过 构造样本实现
        user_sparse_embedding_list, user_dense_value_list = \
            self.input_from_feature_columns(X, self.user_feature_columns, self.embedding_dict)

        user_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list)
        user_embedding = self.user_dnn(user_dnn_input.to(self.device))  # (batch_size, embedding_dim)
        user_embedding = user_embedding.unsqueeze(1)  # (batch, 1, embedding_dim)
        return user_embedding

    def input_from_item_feature_columns(self, X, feature_columns, embedding_dict, support_dense=True):

        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

        if not support_dense and len(dense_feature_columns) > 0:
            raise ValueError(
                "DenseFeat is not supported in dnn_feature_columns")

        sparse_embedding_list = [embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in sparse_feature_columns]

        # 这里返回的就是 movie_id 的 embedding
        if varlen_sparse_feature_columns:
            sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                          varlen_sparse_feature_columns)
            feat_name = varlen_sparse_feature_columns[0].name
            item_embedding = sequence_embed_dict[feat_name]
            # shape is (batch, movie_id_len, feat_dim)

            varlen_sparse_embedding_list = [item_embedding]
        else:
            varlen_sparse_embedding_list = []
        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            dense_feature_columns]

        return sparse_embedding_list + varlen_sparse_embedding_list, dense_value_list

    def compute_input_dim(self, feature_columns, include_sparse=True, include_dense=True, feature_group=False):
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(
            feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        dense_input_dim = sum(
            map(lambda x: x.dimension, dense_feature_columns))
        if feature_group:
            sparse_input_dim = len(sparse_feature_columns)
        else:
            sparse_input_dim = sum(feat.embedding_dim for feat in sparse_feature_columns)
        input_dim = 0
        if include_sparse:
            input_dim += sparse_input_dim
        if include_dense:
            input_dim += dense_input_dim
        return input_dim
