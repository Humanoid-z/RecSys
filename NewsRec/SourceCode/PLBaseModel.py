from abc import abstractmethod
from functools import partial

import torch.nn.functional as F

from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import torch
import torch.nn as nn
from deepctr_torch.inputs import build_input_features
from deepctr_torch.inputs import varlen_embedding_lookup
from deepctr_torch.inputs import SparseFeat, DenseFeat, VarLenSparseFeat
class SequencePoolingLayer(nn.Module):
    """The SequencePoolingLayer is used to apply pooling operation(sum,mean,max) on variable-length sequence feature/multi-value feature.

      Input shape
        - A list of two  tensor [seq_value,seq_len]

        - seq_value is a 3D tensor with shape: ``(batch_size, T, embedding_size)``

        - seq_len is a 2D tensor with shape : ``(batch_size, 1)``,indicate valid length of each sequence.

      Output shape
        - 3D tensor with shape: ``(batch_size, 1, embedding_size)``.

      Arguments
        - **mode**:str.Pooling operation to be used,can be sum,mean or max.

    """

    def __init__(self, mode='mean', supports_masking=False):

        super(SequencePoolingLayer, self).__init__()
        if mode not in ['sum', 'mean', 'max']:
            raise ValueError('parameter mode should in [sum, mean, max]')
        self.supports_masking = supports_masking
        self.mode = mode
        self.eps = torch.FloatTensor([1e-8])


    def _sequence_mask(self, lengths, maxlen=None, dtype=torch.bool):
        # Returns a mask tensor representing the first N positions of each cell.
        if maxlen is None:
            maxlen = lengths.max()
        row_vector = torch.arange(0, maxlen, 1).to(self.eps.device)
        matrix = lengths.unsqueeze(-1).to(self.eps.device)
        mask = row_vector < matrix

        mask.type(dtype)
        return mask

    def forward(self, seq_value_len_list):
        if self.supports_masking:
            uiseq_embed_list, mask = seq_value_len_list  # [B, T, E], [B, 1]
            mask = mask.float()
            user_behavior_length = torch.sum(mask, dim=-1, keepdim=True)
            mask = mask.unsqueeze(2)
        else:
            uiseq_embed_list, user_behavior_length = seq_value_len_list  # [B, T, E], [B, 1]
            mask = self._sequence_mask(user_behavior_length, maxlen=uiseq_embed_list.shape[1],
                                       dtype=torch.float32)  # [B, 1, maxlen]
            mask = torch.transpose(mask, 1, 2)  # [B, maxlen, 1]

        embedding_size = uiseq_embed_list.shape[-1]

        mask = torch.repeat_interleave(mask, embedding_size, dim=2)  # [B, maxlen, E]

        if self.mode == 'max':
            hist = uiseq_embed_list - (1 - mask) * 1e9
            hist = torch.max(hist, dim=1, keepdim=True)[0]
            return hist
        hist = uiseq_embed_list * mask.float().to(uiseq_embed_list.device)
        hist = torch.sum(hist, dim=1, keepdim=False)

        if self.mode == 'mean':
            # self.eps = self.eps.to(user_behavior_length.device)
            hist = torch.div(hist, user_behavior_length.type(torch.float32) + self.eps.to(hist.device))

        hist = torch.unsqueeze(hist, dim=1)
        return hist
def get_varlen_pooling_list(embedding_dict, features, feature_index, varlen_sparse_feature_columns):
    varlen_sparse_embedding_list = []
    for feat in varlen_sparse_feature_columns:
        seq_emb = embedding_dict[feat.name]
        if feat.length_name is None:
            seq_mask = features[:, feature_index[feat.name][0]:feature_index[feat.name][1]].long() != 0

            emb = SequencePoolingLayer(mode=feat.combiner, supports_masking=True)(
                [seq_emb, seq_mask])
        else:
            seq_length = features[:, feature_index[feat.length_name][0]:feature_index[feat.length_name][1]].long()
            emb = SequencePoolingLayer(mode=feat.combiner, supports_masking=False)(
                [seq_emb, seq_length])
        varlen_sparse_embedding_list.append(emb)
    return varlen_sparse_embedding_list
# copy from DeepMatch and remove device
def create_embedding_matrix(feature_columns, init_std=0.001, linear=False, sparse=False):
    # Return nn.ModuleDict: for sparse features, {embedding_name: nn.Embedding}
    # for varlen sparse features, {embedding_name: nn.EmbeddingBag}
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []

    # for feat in sparse_feature_columns:
    #     print(feat.embedding_name)
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if len(feature_columns) else []

    embedding_dict = nn.ModuleDict(  # 重名的不会额外添加
        {feat.embedding_name: nn.Embedding(feat.vocabulary_size, feat.embedding_dim if not linear else 1, sparse=sparse)
         for feat in
         sparse_feature_columns + varlen_sparse_feature_columns}
    )

    # for feat in varlen_sparse_feature_columns:
    #     embedding_dict[feat.embedding_name] = nn.EmbeddingBag(
    #         feat.dimension, embedding_size, sparse=sparse, mode=feat.combiner)

    for tensor in embedding_dict.values():
        nn.init.normal_(tensor.weight, mean=0, std=init_std)

    return embedding_dict
    

# 使用 Pl 作为 fit 的接口
class PLBaseModel(LightningModule):
    """Base class for all DeepMatch_Torch models.
    This model inspired from: https://github.com/Rose-STL-Lab/torchTS/blob/main/torchts/nn/model.py
    Args:
        optimizer (torch.optim.Optimizer): Optimizer
        opimizer_args (dict): Arguments for the optimizer
        criterion: Loss function
        criterion_args (dict): Arguments for the loss function
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler
        scheduler_args (dict): Arguments for the scheduler
        scaler (torchts.utils.scaler.Scaler): Scaler
    """

    def __init__(
        self,
        user_feature_columns,
        item_feature_columns,
        optimizer=None,
        optimizer_args=None,
        # criterion=F.mse_loss,
        criterion=F.binary_cross_entropy,
        criterion_args=None,
        scheduler=None,
        scheduler_args=None,
        scaler=None,
        config={},
        **kwargs,
    ):
        super().__init__()
        self.user_feature_columns = user_feature_columns
        self.item_feature_columns = item_feature_columns
        # DeepMatch side 用于增加兼容
        self.config = config  # 增加 config 用于设置 DeepMatch 侧需要的参数
        # 优先使用 kwargs 的配置
        # TODO: 以后要消灭所有的 device , 
        self.config.update(kwargs)
        
        # 用于判断模型 输出 logits 还是 user/item 的 vector 表示
        self.mode = self.config.get('mode', 'train')

        self.linear_feature_columns = user_feature_columns + item_feature_columns
        self.dnn_feature_columns = self.linear_feature_columns 

        # 在 pl 中不需要 to(device)
        self.reg_loss = torch.zeros((1,))
        self.aux_loss = torch.zeros((1,))

        self.feature_index = build_input_features(
            self.linear_feature_columns)

        self.embedding_dict = create_embedding_matrix(self.dnn_feature_columns, 
                    self.config.get("init_std"), sparse=False, )


        self.regularization_weight = []

        self.add_regularization_weight(self.embedding_dict.parameters(), 
            l2=self.config.get("l2_reg_embedding"))


        self.criterion = criterion
        self.criterion_args = criterion_args
        self.scaler = scaler

        # 增加了选择 optimizer 的方式
        optimizer = self.init_optimizer(optimizer)
        if optimizer_args is not None:
            self.optimizer = partial(optimizer, **optimizer_args)
        else:
            self.optimizer = optimizer

        if scheduler is not None and scheduler_args is not None:
            self.scheduler = partial(scheduler, **scheduler_args)
        else:
            self.scheduler = scheduler

    def fit(self, x, y, max_epochs=10, batch_size=128):
        """Fits model to the given data.
        Args:
            x (torch.Tensor): Input data
            y (torch.Tensor): Output data
            max_epochs (int): Number of training epochs
            batch_size (int): Batch size for torch.utils.data.DataLoader
        """
        x, y = self.construct_input(x, y)
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        trainer = Trainer(max_epochs=max_epochs, accelerator="auto")
        trainer.fit(self, loader)

    def prepare_batch(self, batch):
        return batch

    def _step(self, batch, batch_idx, num_batches):
        """
        Args:
            batch: Output of the torch.utils.data.DataLoader
            batch_idx: Integer displaying index of this batch
            dataset: Data set to use
        Returns: loss for the batch
        """
        x, y = self.prepare_batch(batch)

        if self.training:
            batches_seen = batch_idx + self.current_epoch * num_batches
        else:
            batches_seen = batch_idx

        pred = self(x)

        if self.scaler is not None:
            y = self.scaler.inverse_transform(y)
            pred = self.scaler.inverse_transform(pred)

        # 假设是 softmax, 应该把 y 设置为 Long类型
        if self.criterion == F.cross_entropy:
            y = y.long()
        y = y.unsqueeze(-1)

        loss = self.criterion(pred, y)


        return loss

    def training_step(self, batch, batch_idx):
        """Trains model for one step.
        Args:
            batch (torch.Tensor): Output of the torch.utils.data.DataLoader
            batch_idx (int): Integer displaying index of this batch
        """
        train_loss = self._step(batch, batch_idx, len(self.trainer.train_dataloader))
        self.log(
            "train_loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return train_loss

    def validation_step(self, batch, batch_idx):
        """Validates model for one step.
        Args:
            batch (torch.Tensor): Output of the torch.utils.data.DataLoader
            batch_idx (int): Integer displaying index of this batch
        """
        val_loss = self._step(batch, batch_idx, len(self.trainer.val_dataloader))
        self.log("val_loss", val_loss)
        return val_loss

    def test_step(self, batch, batch_idx):
        """Tests model for one step.
        Args:
            batch (torch.Tensor): Output of the torch.utils.data.DataLoader
            batch_idx (int): Integer displaying index of this batch
        """
        test_loss = self._step(batch, batch_idx, len(self.trainer.test_dataloader))
        self.log("test_loss", test_loss)
        return test_loss

    @abstractmethod
    def forward(self, x):
        """Forward pass.
        Args:
            x (torch.Tensor): Input data
        Returns:
            torch.Tensor: Predicted data
        """

    def predict(self, x):
        """Runs model inference.
        Args:
            x (torch.Tensor): Input data
        Returns:
            torch.Tensor: Predicted data
        """
        return self(x).detach()

    def configure_optimizers(self):
        """Configure optimizer.
        Returns:
            torch.optim.Optimizer: Optimizer
        """
        optimizer = self.optimizer(self.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer)
            return [optimizer], [scheduler]

        return optimizer

    def init_optimizer(self, optimizer_or_name):
        """init_optimizer 假设 optimizer 是一个string, 那么返回 torch.optim对应的 optimizer
            否则，返回自己
        Args:
            optimizer_or_name ([type]): [description]

        Returns:
            [type]: [description]
        """
        if isinstance(optimizer_or_name, str):
            if optimizer_or_name == "Adam":
                return torch.optim.Adam
            elif optimizer_or_name == "SGD":
                return torch.optim.SGD
        else:
            return optimizer_or_name
            
    def construct_input(self, x, y):
        # 有可能直接输出的就是 tensor
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
        
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)
        x = torch.from_numpy(np.concatenate(x, axis=-1))
        y = torch.tensor(y, dtype=torch.float32)
        return x, y

    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        # For a Parameter, put it in a list to keep Compatible with get_regularization_loss()
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        # For generators, filters and ParameterLists, convert them to a list of tensors to avoid bugs.
        # e.g., we can't pickle generator objects when we save the model.
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))

    def get_regularization_loss(self, ):
        total_reg_loss = torch.zeros((1,), )
        for weight_list, l1, l2 in self.regularization_weight:
            for w in weight_list:
                if isinstance(w, tuple):
                    parameter = w[1]  # named_parameters
                else:
                    parameter = w
                if l1 > 0:
                    total_reg_loss += torch.sum(l1 * torch.abs(parameter))
                if l2 > 0:
                    try:
                        total_reg_loss += torch.sum(l2 * torch.square(parameter))
                    except AttributeError:
                        total_reg_loss += torch.sum(l2 * parameter * parameter)

        return total_reg_loss
    
    def input_from_feature_columns(self, X, feature_columns, embedding_dict, support_dense=True):
        
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
        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                      varlen_sparse_feature_columns)
        varlen_sparse_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                               varlen_sparse_feature_columns)
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

    def rebuild_feature_index(self, feature_columns):
        # 为了满足 单独预测 user/item vector 的需求，需要重新知道 feature_columns 的位置
        self.feature_index = build_input_features(feature_columns)
        return self

    def full_predict(self, x, batch_size=256):
        # 有可能直接输入的就是 tensor
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
        
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)
        x = torch.from_numpy(np.concatenate(x, axis=-1))
        dataset = TensorDataset(x)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        ret = []
        for batch in loader:            
            tmp_result = self.predict(batch[0])
            ret.append(tmp_result)
        return torch.cat(ret, axis=0)
        
