from __future__ import print_function, absolute_import, division

import math
import sys
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle

from pcdc_imvc.utils import clustering
from pcdc_imvc.utils.next_batch import next_batch, next_batch_multiview


def compute_joint(x_out, x_tf_out):
    bn, k = x_out.size()
    assert x_tf_out.size(0) == bn and x_tf_out.size(1) == k

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)
    p_i_j = p_i_j.sum(dim=0)
    p_i_j = (p_i_j + p_i_j.t()) / 2.0
    p_i_j = p_i_j / p_i_j.sum()

    return p_i_j


def maximize_mutual_information_loss(x_out, x_tf_out, lamb=1.0, eps=sys.float_info.epsilon):
    _, k = x_out.size()
    p_i_j = compute_joint(x_out, x_tf_out)
    assert p_i_j.size() == (k, k)

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

    p_i_j = torch.where(p_i_j < eps, torch.tensor([eps], device=p_i_j.device), p_i_j)
    p_j = torch.where(p_j < eps, torch.tensor([eps], device=p_j.device), p_j)
    p_i = torch.where(p_i < eps, torch.tensor([eps], device=p_i.device), p_i)

    loss = -p_i_j * (torch.log(p_i_j) - lamb * torch.log(p_j) - lamb * torch.log(p_i))
    return loss.sum()


def instance_contrastive_Loss(x_out, x_tf_out, lamb=1.0, EPS=sys.float_info.epsilon):
    # Backward-compatible alias.
    return maximize_mutual_information_loss(x_out, x_tf_out, lamb=lamb, eps=EPS)

class Autoencoder(nn.Module):
    """AutoEncoder module that projects features to latent space."""
    # 维度约定：
    # - B: batch size（批大小）
    # - D: 输入特征维度（每个视图的原始特征维）
    # - L: 潜表示维度（encoder_dim 的最后一个元素）
    # 数据流：x [B, D] -> encoder -> latent z [B, L] (Softmax 归一化到概率分布) -> decoder -> x_hat [B, D]

    def __init__(self,
                 encoder_dim,
                 activation='relu',
                 batchnorm=True,
                 attention_cfg=None,         # 新增：注意力配置（可为 None）--------------------------------
                 decoder_attention=False     # 新增：解码器是否也使用注意力---------------------------------
                 ):
        """Constructor.

        Args:
          encoder_dim: Should be a list of ints, hidden sizes of
            encoder network, the last element is the size of the latent representation.
          activation: Including "sigmoid", "tanh", "relu", "leakyrelu". We recommend to
            simply choose relu.
          batchnorm: if provided should be a bool type. It provided whether to use the
            batchnorm in autoencoders.
        """
        super(Autoencoder, self).__init__()
        self._dim = len(encoder_dim) - 1  #encoder_dim 是一个列表，按顺序给出编码器各层的维度，最后一个是潜表示维度 L；列表有 n 个维度，就有 n-1 次“维度跃迁”（线性层数），因此 _dim = n-1
        self._activation = activation
        self._batchnorm = batchnorm

        # 解析注意力配置（默认关闭）----------------------------------------------------------------------------------
        att_enable = bool(attention_cfg.get('enable', False)) if attention_cfg else False
        att_type = str(attention_cfg.get('type', 'se')) if attention_cfg else 'se'
        att_reduction = int(attention_cfg.get('reduction', 8)) if attention_cfg else 8
        att_residual = bool(attention_cfg.get('residual', True)) if attention_cfg else True
        att_scale = float(attention_cfg.get('scale', 0.1)) if attention_cfg else 0.1
        att_place = str(attention_cfg.get('place', 'pre_bn')) if attention_cfg else 'pre_bn'  # 'pre_bn'|'post_act'
        att_apply_last = bool(attention_cfg.get('apply_last', False)) if attention_cfg else False
        att_apply_indices = attention_cfg.get('apply_indices', None) if attention_cfg else None
        att_only_1024 = bool(attention_cfg.get('only_1024', True)) if attention_cfg else True
        eca_gamma = float(attention_cfg.get('eca_gamma', 2.0)) if attention_cfg else 2.0
        eca_b = float(attention_cfg.get('eca_b', 1.0)) if attention_cfg else 1.0
        if att_apply_indices is not None:
            att_apply_indices = set(int(i) for i in att_apply_indices)
        #------------------------------------
        """
        encoder_layers = []#结构：Linear → BatchNorm1d → 激活函数
        for i in range(self._dim): #代码用 _dim 控制循环，构建 n-1 个 Linear 层；其中最后一层不再加 BN/激活，而是单独在末端接 Softmax 作为潜表示输出。
            encoder_layers.append(
                nn.Linear(encoder_dim[i], encoder_dim[i + 1]))#添加一个线性层（全连接层），其输入维度为 encoder_dim[i]，输出维度为 encoder_dim[i + 1]。
            if i < self._dim - 1: #条件 i < self._dim - 1：只对中间隐藏层添加归一化与激活；最后一层不加激活/BN
                if self._batchnorm:
                    encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1]))
                if self._activation == 'sigmoid':
                    encoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    encoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    encoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
                # 插入轻量注意力（仅隐藏层）--------------------------------------------------------------------------------------------------
                if att_enable:
                    encoder_layers.append(FeatureSE(encoder_dim[i + 1], reduction=att_reduction, residual=att_residual))#-------------------

        # 编码器末端使用 Softmax(dim=1)，得到每个样本的概率型潜表示 z ∈ R^{B×L}，每行非负且和为 1。
        encoder_layers.append(nn.Softmax(dim=1))
        """
        def _use_att(i, out_c):
            if not att_enable:
                return False
            ok_by_pos = (att_apply_last or i < self._dim - 2)
            ok_by_list = (att_apply_indices is None or i in att_apply_indices)
            ok_by_dim = (not att_only_1024) or out_c == 1024
            return ok_by_pos and ok_by_list and ok_by_dim

        encoder_layers = []#----------------------------------------------------------------------------
        for i in range(self._dim):
            in_c, out_c = encoder_dim[i], encoder_dim[i + 1]
            encoder_layers.append(nn.Linear(in_c, out_c))

            if i < self._dim - 1:
                use_att = _use_att(i, out_c)
                if att_place == 'pre_bn':
                    if use_att:
                        if att_type == 'eca':
                            encoder_layers.append(FeatureECA(out_c, gamma=eca_gamma, b=eca_b))
                        else:
                            encoder_layers.append(FeatureSE(out_c, reduction=att_reduction, residual=att_residual, scale=att_scale))
                    if self._batchnorm:
                        encoder_layers.append(nn.BatchNorm1d(out_c))
                    if self._activation == 'sigmoid':
                        encoder_layers.append(nn.Sigmoid())
                    elif self._activation == 'leakyrelu':
                        encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                    elif self._activation == 'tanh':
                        encoder_layers.append(nn.Tanh())
                    elif self._activation == 'relu':
                        encoder_layers.append(nn.ReLU())
                    else:
                        raise ValueError('Unknown activation type %s' % self._activation)
                else:  # post_act
                    if self._batchnorm:
                        encoder_layers.append(nn.BatchNorm1d(out_c))
                    if self._activation == 'sigmoid':
                        encoder_layers.append(nn.Sigmoid())
                    elif self._activation == 'leakyrelu':
                        encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                    elif self._activation == 'tanh':
                        encoder_layers.append(nn.Tanh())
                    elif self._activation == 'relu':
                        encoder_layers.append(nn.ReLU())
                    else:
                        raise ValueError('Unknown activation type %s' % self._activation)
                    if use_att:
                        if att_type == 'eca':
                            encoder_layers.append(FeatureECA(out_c, gamma=eca_gamma, b=eca_b))
                        else:
                            encoder_layers.append(FeatureSE(out_c, reduction=att_reduction, residual=att_residual, scale=att_scale))
            else:
                # 末端保持 Softmax（你的现有实现）
                encoder_layers.append(nn.Softmax(dim=1))#---------------------------------------------------------------------

        self._encoder = nn.Sequential(*encoder_layers) #nn.Sequential(*encoder_layers) 会把列表里的层按顺序串起来，self._encoder 就是整个编码器网络，可以直接用在前向传播里。                                             
        
        decoder_dim = [i for i in reversed(encoder_dim)]#将 encoder_dim 列表反转，得到解码器每一层的输入输出维度。
        decoder_layers = []
        for i in range(self._dim):
            decoder_layers.append(
                nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            if self._batchnorm:
                decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
            if self._activation == 'sigmoid':
                decoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                decoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                decoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
            # 解码器注意力（可选，默认关闭）------------------------------------------------------------------------------------------------
            if decoder_attention and i < self._dim - 1 and att_enable:
                if att_type == 'eca':
                    decoder_layers.append(FeatureECA(decoder_dim[i + 1], gamma=eca_gamma, b=eca_b))
                else:
                    decoder_layers.append(FeatureSE(decoder_dim[i + 1],
                                                    reduction=att_reduction,
                                                    residual=att_residual,
                                                    scale=att_scale))#--------------------------------

        self._decoder = nn.Sequential(*decoder_layers)

    def encoder(self, x):
        """Encode sample features.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [n_nodes, latent_dim] float tensor, representation Z.
        """
                # 输入 x 形状：[B, D]，输出 latent 形状：[B, L]
        latent = self._encoder(x)
        return latent
    def decoder(self, latent):
        """Decode sample features.

            Args:
              latent: [num, latent_dim] float tensor, representation Z.

            Returns:
              x_hat: [n_nodes, feat_dim] float tensor, reconstruction x.
        """
                # 输入 latent 形状：[B, L]，输出 x_hat 形状：[B, D]
        x_hat = self._decoder(latent)
        return x_hat
    def forward(self, x):
        """Pass through autoencoder.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [num, latent_dim] float tensor, representation Z.
              x_hat:  [num, feat_dim] float tensor, reconstruction x.
        """
                # 前向：x [B, D] -> latent [B, L] -> x_hat [B, D]
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        return x_hat, latent 

class PrototypeConstraintQuantizer(nn.Module):
    def __init__(self, d, K=64, tau=1.0, hard_eval=True):
        super().__init__()
        self.codebook = nn.Parameter(torch.randn(K, d) * 0.1)
        self.tau = float(tau)
        self.hard_eval = bool(hard_eval)

    def forward(self, h):
        # h: [B, d]
        e = self.codebook  # [K, d]
        h2 = (h ** 2).sum(dim=1, keepdim=True)           # [B, 1]
        e2 = (e ** 2).sum(dim=1, keepdim=True).t()       # [1, K]
        he = h @ e.t()                                   # [B, K]
        logits = -(h2 + e2 - 2 * he)                     # 负欧氏距离作为相似度

        if self.training:
            g = -torch.log(-torch.log(torch.rand_like(logits).clamp_min(1e-9)))
            a = torch.softmax((logits + g) / max(1e-6, self.tau), dim=1)  # 软分配
        else:
            if self.hard_eval:
                idx = torch.argmax(logits, dim=1)                           # 硬分配
                a = F.one_hot(idx, num_classes=e.shape[0]).float()
            else:
                a = torch.softmax(logits / max(1e-6, self.tau), dim=1)      # 软分配

        q = a @ e  # [B, d]
        return q
        
class CompletionModule(nn.Module):
    """Dual Inference module that projects features from corresponding latent space."""
        # 维度约定：输入/输出均在潜空间 L 上
        # 数据流：z_src [B, L] -> encoder -> h [B, *] -> decoder(末端 Softmax) -> z_tgt_hat [B, L]
        # 用于跨视图潜表示恢复（与目标视图的潜表示做 MSE 一致性约束）。

    def __init__(self,
                 inference_dim,
                 activation='relu',
                 batchnorm=True,
                 diveq_cfg=None):
        """Constructor.

        Args:
          inference_dim: Should be a list of ints, hidden sizes of
            inference network, the last element is the size of the latent representation of autoencoder.
          activation: Including "sigmoid", "tanh", "relu", "leakyrelu". We recommend to
            simply choose relu.
          batchnorm: if provided should be a bool type. It provided whether to use the
            batchnorm in autoencoders.
        """
        super(CompletionModule, self).__init__()

        # inference_dim 形如 [L, h1, ..., L]；_depth 表示线性“跃迁”次数（层数-1）
        self._depth = len(inference_dim) - 1
        self._activation = activation
        self._inference_dim = inference_dim

        encoder_layers = []
        for i in range(self._depth):
            # 逐层将源视图潜表示 z_src 投到隐藏空间；可选 BN+激活
            encoder_layers.append(
                nn.Linear(self._inference_dim[i], self._inference_dim[i + 1]))
            if batchnorm:
                encoder_layers.append(nn.BatchNorm1d(self._inference_dim[i + 1]))
            if self._activation == 'sigmoid':
                encoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                encoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                encoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(self._depth, 0, -1):
            # 对称解码：把隐藏表示映射回目标视图潜空间
            decoder_layers.append(
                nn.Linear(self._inference_dim[i], self._inference_dim[i - 1]))
            if i > 1:
                if batchnorm:
                    decoder_layers.append(nn.BatchNorm1d(self._inference_dim[i - 1]))
                if self._activation == 'sigmoid':
                    decoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    decoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    decoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        # 末端 Softmax：输出目标视图潜表示的概率型估计 z_tgt_hat ∈ R^{B×L}
        decoder_layers.append(nn.Softmax(dim=1))
        self._decoder = nn.Sequential(*decoder_layers)
        # === DIVEQ 开关 ===
        self._use_diveq = bool(diveq_cfg.get('enable', True)) if diveq_cfg else True
        if self._use_diveq:
            d = self._inference_dim[-1]
            K = int(diveq_cfg.get('K', 64))
            tau = float(diveq_cfg.get('tau', 1.0))
            hard_eval = bool(diveq_cfg.get('hard_eval', True))
            self._diveq = PrototypeConstraintQuantizer(d=d, K=K, tau=tau, hard_eval=hard_eval)
        else:
            self._diveq = None
    def forward(self, x):
        """Data recovery by inference.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [num, latent_dim] float tensor.
              output:  [num, feat_dim] float tensor, recovered data.
        """
          # x 为源视图潜表示，encoder 提取跨视图共享表示，decoder 估计目标视图潜表示
        latent = self._encoder(x)
        if self._diveq is not None and self._use_diveq:
            q = self._diveq(latent)
            output = self._decoder(q)
        else:
            output = self._decoder(latent)
        return output, latent

def target_l2(q):
    # 将每个样本向量做元素平方后按行归一化：q^2 / sum(q^2)
    # 输入 q：[B, L]，输出同形状 [B, L]
    return ((q ** 2).t() / (q ** 2).sum(1)).t()

def siamese_similarity_contrastive_loss(p1, p2, z1_stop, z2_stop, eps=1e-8):
    # p1 对齐 z2_stop，p2 对齐 z1_stop；z1_stop/z2_stop 需提前 detach
    p1 = F.normalize(p1, dim=1, eps=eps)
    p2 = F.normalize(p2, dim=1, eps=eps)
    z1 = F.normalize(z1_stop, dim=1, eps=eps)
    z2 = F.normalize(z2_stop, dim=1, eps=eps)
    loss = 2 - ((p1 * z2).sum(dim=1) + (p2 * z1).sum(dim=1))
    return loss.mean()


def simsiam_align(p1, p2, z1_stop, z2_stop, eps=1e-8):
    # Backward-compatible alias.
    return siamese_similarity_contrastive_loss(p1, p2, z1_stop, z2_stop, eps=eps)

def soft_ce_loss(pred_prob, target_prob, eps=1e-8):# === 新增：软标签交叉熵（分布对分布） ===----------------------修改部分4-----------------------
    # pred_prob/target_prob: [B, L]，均为概率分布
    pred_prob = pred_prob.clamp_min(eps)
    target_prob = target_prob.clamp_min(eps)
    return (-target_prob * pred_prob.log()).sum(dim=1).mean()# === 新增：软标签交叉熵（分布对分布） ===----------------------修改部分4-----------------------

class PCDC(nn.Module):

    def __init__(self, config):
        """Constructor.

        Args:
            config: parameters defined in configure.py.
        """
        super(PCDC, self).__init__()
        self._config = config

        # 读取注意力配置（兼容旧配置，默认关闭）------------------------------------------------------------------------------------------
        att_cfg = config['Autoencoder'].get('attention', {'enable': False})
        dec_att = bool(att_cfg.get('decoder', False))#-------------------------------------------------------------------------------

        #比较配置里两个 autoencoder 架构列表的最后一项（arch1[-1]、arch2[-1]），即两视图的潜表示维度 L 是否相同
        if self._config['Autoencoder']['arch1'][-1] != self._config['Autoencoder']['arch2'][-1]:
            raise ValueError('Inconsistent latent dim!')

        # 潜表示维度 L（两视图保持一致）
        self._latent_dim = config['Autoencoder']['arch1'][-1]
        #----------------------------------------修改2--------------------------------------------
        # 轻量温度系数（可不配，默认 1.0）
        #self._tau = config['training'].get('tau_sim', 1.0)
        #----------------------------------------修改2--------------------------------------------


        # Completion 模块维度：输入为 L，后接 Completion/Inference 配置的隐藏层
        completion_cfg = self._config.get('Completion', self._config['Inference'])
        self._dims_view1 = [self._latent_dim] + completion_cfg['arch1']
        self._dims_view2 = [self._latent_dim] + completion_cfg['arch2']

        # View-specific autoencoders
        # AE1: 输入 D1 -> ... -> L -> ... -> D1
        self.autoencoder1 = Autoencoder(config['Autoencoder']['arch1'], 
                                        config['Autoencoder']['activations1'],
                                        config['Autoencoder']['batchnorm'],
                                        attention_cfg=att_cfg,
                                        decoder_attention=dec_att)
        # AE2: 输入 D2 -> ... -> L -> ... -> D2
        self.autoencoder2 = Autoencoder(config['Autoencoder']['arch2'], 
                                        config['Autoencoder']['activations2'],
                                        config['Autoencoder']['batchnorm'],
                                        attention_cfg=att_cfg,
                                        decoder_attention=dec_att)

        # ...existing code...

        # 跨视图 Completion，一致性训练与评估填补缺失视图时沿用同一流程
        # img2txt: z_img [B, L] -> ... -> z_txt_hat [B, L]
        diveq_cfg = config.get('training', {}).get(
            'prototype_constraint',
            config.get('training', {}).get('diveq', {'enable': False}),
        )
        self.img2txt = CompletionModule(self._dims_view1, activation='relu', batchnorm=True, diveq_cfg=diveq_cfg)
        # txt2img: z_txt [B, L] -> ... -> z_img_hat [B, L]
        self.txt2img = CompletionModule(self._dims_view2, activation='relu', batchnorm=True, diveq_cfg=diveq_cfg)

        
        # 轻量预测头（SimSiam），保持维度不变，便于回退
        self.predictor1 = nn.Sequential(
            nn.Linear(self._latent_dim, self._latent_dim),
            nn.BatchNorm1d(self._latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self._latent_dim, self._latent_dim)
        )
        self.predictor2 = nn.Sequential(
            nn.Linear(self._latent_dim, self._latent_dim),
            nn.BatchNorm1d(self._latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self._latent_dim, self._latent_dim)
        )
        self.best_latent_fusion = None
        self.best_epoch = None
        
    def train(self, config, logger, accumulated_metrics, x1_train,x2_train, Y_list, mask, optimizer, device):
        """Training the model.

            Args:
              config: parameters which defined in configure.py.
              logger: print the information
              accumulated_metrics: list of metrics
              x1_train: data of view 1
              x2_train: data of view 2
              Y_list: labels
              mask: generate missing data
              optimizer: adam is used in our experiments
              device: to cuda if gpu is used
                        Returns:
                            clustering performance: acc, nmi, ari, fscore

        """

        self.to(device)
        epochs_total = config['training']['epoch']
        batch_size = config['training']['batch_size']
        classes=config['training']['class_num']
        best_acc, best_nmi, best_ari, best_fscore = 0.0, 0.0, 0.0, 0.0
        if 'fscore' not in accumulated_metrics:
            accumulated_metrics['fscore'] = []
        # Get complete data for training
        # mask 形状：[N, 2]，mask[i, j]==1 表示样本 i 在视图 j 可用
        # flag：选出两视图都不缺失的样本（用于有监督的对齐与重构）
        flag = (torch.LongTensor([1, 1]).to(device) == mask).int()
        flag = (flag[:, 1] + flag[:, 0]) == 2
        # 训练用完整样本：
        # train_view1 形状：[N_full, D1]，train_view2 形状：[N_full, D2]
        train_view1 = x1_train[flag].to(device).float()  # 确保设备一致
        train_view2 = x2_train[flag].to(device).float()

        def _mse_per_sample(a, b):
            return ((a - b) ** 2).mean(dim=1)

        def _inf_loss(pred, target, use_conf=False, conf=None, loss_type='mse'):
            if loss_type == 'mse':
                per = _mse_per_sample(pred, target)
            elif loss_type == 'soft_ce':
                per = -(target.clamp_min(1e-8) * pred.clamp_min(1e-8).log()).sum(dim=1)
            else:
                raise ValueError(f"Unknown inf_loss: {loss_type}")
            if use_conf and conf is not None:
                per = per * conf.detach().clamp_min(1e-6)
            return per.mean() if per.numel() > 0 else torch.tensor(0.0, device=device)

        use_conf = config['training'].get('inf_conf_weight', False)
        conf_thresh = float(config['training'].get('conf_thresh', 0.7))

        start_completion = int(config['training'].get('start_completion', config['training'].get('start_inference', 0)))

        for k in range(epochs_total):
            # 对齐打乱两个视图的完整样本（保持索引一致）
            X1, X2 = shuffle(train_view1, train_view2)
            # 逐 epoch 累计各类损失：all0 总损失；all1/2 为两个视图的重构；
            # map1/2 为跨视图推理一致性；all_icl1/2 为对齐/对比分项；all_icl 为对齐总和
            all0 = 0.0
            all1 = 0.0
            all2 = 0.0
            map1 = 0.0
            map2 = 0.0
            all_icl1 = 0.0
            all_icl2 = 0.0
            all_icl = 0.0
            for batch_x1, batch_x2, batch_No in next_batch(X1, X2, batch_size):
                # 批数据：batch_x1 [B, D1]，batch_x2 [B, D2]
                z_half1 = self.autoencoder1.encoder(batch_x1)
                z_half2 = self.autoencoder2.encoder(batch_x2)
                z_half1 = z_half1.to(device).float()
                z_half2 = z_half2.to(device).float()

                # ...existing code...

                # Within-view Reconstruction Loss
                recon1 = F.mse_loss(self.autoencoder1.decoder(z_half1), batch_x1)
                recon2 = F.mse_loss(self.autoencoder2.decoder(z_half2), batch_x2)
                reconstruction_loss = recon1 + recon2
                # Cross-view Contrastive_Loss
                z_1, z_2 = z_half1, z_half2

                # Siamese 对齐
                p1 = self.predictor1(z_1)
                p2 = self.predictor2(z_2)
                loss_icl1 = siamese_similarity_contrastive_loss(p1, p2, z_1.detach(), z_2.detach())

                # 最大化互信息
                loss_icl2 = maximize_mutual_information_loss(z_1, z_2, config['training']['alpha'])
                # 统一为多视图语义：lambda2=互信息, lambda3=Siamese 对齐
                loss_icl = loss_icl2 * config['training']['lambda2'] + loss_icl1 * config['training']['lambda3']
                # Cross-view Completion Loss（直推 + 循环一致性 + 置信度）
                img2txt, _ = self.img2txt(z_half1)
                txt2img, _ = self.txt2img(z_half2)

                inf_loss_type = config['training'].get('inf_loss', 'mse')
                z1_t = z_half1.detach()
                z2_t = z_half2.detach()
                if config['training'].get('inf_sharpen', False):
                    z1_t = target_l2(z1_t)
                    z2_t = target_l2(z2_t)

                direct_loss = torch.tensor(0.0, device=device)
                cycle_loss = torch.tensor(0.0, device=device)
                count_direct = 0
                count_cycle = 0

                # 直推：img -> txt，对齐 z2
                if use_conf:
                    conf12 = F.softmax(img2txt, dim=1).max(dim=1).values
                    valid12 = conf12 > conf_thresh
                    if valid12.sum() != 0:
                        direct_loss += _inf_loss(img2txt[valid12], z2_t[valid12], True, conf12[valid12], inf_loss_type)
                        count_direct += 1
                else:
                    direct_loss += _inf_loss(img2txt, z2_t, False, None, inf_loss_type)
                    count_direct += 1

                # 直推：txt -> img，对齐 z1
                if use_conf:
                    conf21 = F.softmax(txt2img, dim=1).max(dim=1).values
                    valid21 = conf21 > conf_thresh
                    if valid21.sum() != 0:
                        direct_loss += _inf_loss(txt2img[valid21], z1_t[valid21], True, conf21[valid21], inf_loss_type)
                        count_direct += 1
                else:
                    direct_loss += _inf_loss(txt2img, z1_t, False, None, inf_loss_type)
                    count_direct += 1

                # 循环：img -> txt -> img，对齐 z1
                img_cycle, _ = self.txt2img(img2txt)
                if use_conf:
                    conf12 = F.softmax(img2txt, dim=1).max(dim=1).values
                    valid12 = conf12 > conf_thresh
                    if valid12.sum() != 0:
                        cycle_loss += _inf_loss(img_cycle[valid12], z1_t[valid12], True, conf12[valid12], inf_loss_type)
                        count_cycle += 1
                else:
                    cycle_loss += _inf_loss(img_cycle, z1_t, False, None, inf_loss_type)
                    count_cycle += 1

                # 循环：txt -> img -> txt，对齐 z2
                txt_cycle, _ = self.img2txt(txt2img)
                if use_conf:
                    conf21 = F.softmax(txt2img, dim=1).max(dim=1).values
                    valid21 = conf21 > conf_thresh
                    if valid21.sum() != 0:
                        cycle_loss += _inf_loss(txt_cycle[valid21], z2_t[valid21], True, conf21[valid21], inf_loss_type)
                        count_cycle += 1
                else:
                    cycle_loss += _inf_loss(txt_cycle, z2_t, False, None, inf_loss_type)
                    count_cycle += 1

                if count_direct > 0:
                    direct_loss = direct_loss / count_direct
                if count_cycle > 0:
                    cycle_loss = cycle_loss / count_cycle

                completion_loss = direct_loss + cycle_loss
                recon3 = direct_loss
                recon4 = cycle_loss

                # 总损失
                all_loss = loss_icl + reconstruction_loss * config['training']['lambda1']

                if k >= start_completion:
                    lambda4_warm = config['training'].get('lambda4_warmup', 1)
                    lambda4_curr = config['training']['lambda4'] * min(1.0, (k + 1) / max(1, lambda4_warm))
                    all_loss += lambda4_curr * completion_loss

                optimizer.zero_grad()
                all_loss.backward()
                optimizer.step()

                all0 += all_loss.item()
                all1 += recon1.item()
                all2 += recon2.item()
                map1 += recon3.item()
                map2 += recon4.item()
                all_icl1 += loss_icl1.item()
                all_icl2 += loss_icl2.item()
                all_icl += loss_icl.item()
            eval_now = ((k + 1) == 1) or (((k + 1) % config['print_num']) == 0)

            # evalution
            if eval_now:
                with torch.no_grad():
                    self.autoencoder1.eval(), self.autoencoder2.eval()
                    self.img2txt.eval(), self.txt2img.eval()

                    # 评估阶段：根据 mask 构建完整与缺失索引
                    img_idx_eval = mask[:, 0] == 1  # 图像视图完整样本
                    txt_idx_eval = mask[:, 1] == 1
                    img_missing_idx_eval = mask[:, 0] == 0 # 图像视图缺失样本
                    txt_missing_idx_eval = mask[:, 1] == 0

                    # 选择核心视图：可配置（默认关闭，固定为视图0）
                    use_core_by_missing = config['training'].get('core_by_missing', False)
                    if use_core_by_missing:
                        view_avail = mask.sum(dim=0)
                        core_view = int(torch.argmax(view_avail).item())
                    else:
                        core_view = 0

                    # 对各自视图的完整部分进行编码：得到潜表示 [*, L]
                    imgs_latent_eval = self.autoencoder1.encoder(x1_train[img_idx_eval])
                    txts_latent_eval = self.autoencoder2.encoder(x2_train[txt_idx_eval])

                    # 构建全体样本的潜表示占位：两个视图各 [N, L]
                    latent_code_img_eval = torch.zeros(x1_train.shape[0], config['Autoencoder']['arch1'][-1]).to(device)
                    latent_code_txt_eval = torch.zeros(x2_train.shape[0], config['Autoencoder']['arch2'][-1]).to(device)

                    if core_view == 0:
                        # 核心=图像视图：图像缺失用文本推断；文本缺失用图像推断
                        core_missing = img_missing_idx_eval
                        other_has = core_missing & txt_idx_eval
                        if other_has.sum() != 0:
                            other_latent = self.autoencoder2.encoder(x2_train[other_has])
                            core_recon, _ = self.txt2img(other_latent)
                            latent_code_img_eval[other_has] = core_recon

                        noncore_missing = txt_missing_idx_eval
                        core_has = noncore_missing & img_idx_eval
                        if core_has.sum() != 0:
                            core_latent = self.autoencoder1.encoder(x1_train[core_has])
                            noncore_recon, _ = self.img2txt(core_latent)
                            latent_code_txt_eval[core_has] = noncore_recon
                    else:
                        # 核心=文本视图：文本缺失用图像推断；图像缺失用文本推断
                        core_missing = txt_missing_idx_eval
                        other_has = core_missing & img_idx_eval
                        if other_has.sum() != 0:
                            other_latent = self.autoencoder1.encoder(x1_train[other_has])
                            core_recon, _ = self.img2txt(other_latent)
                            latent_code_txt_eval[other_has] = core_recon

                        noncore_missing = img_missing_idx_eval
                        core_has = noncore_missing & txt_idx_eval
                        if core_has.sum() != 0:
                            core_latent = self.autoencoder2.encoder(x2_train[core_has])
                            noncore_recon, _ = self.txt2img(core_latent)
                            latent_code_img_eval[core_has] = noncore_recon

                    # 填充完整样本的潜表示
                    latent_code_img_eval[img_idx_eval] = imgs_latent_eval
                    latent_code_txt_eval[txt_idx_eval] = txts_latent_eval
                    # 两视图潜表示按特征维拼接：latent_fusion [N, 2L]
                    latent_fusion = torch.cat([latent_code_img_eval, latent_code_txt_eval], dim=1).cpu().numpy()

                    scores = clustering.get_score([latent_fusion], Y_list,
                                                  accumulated_metrics['acc'],
                                                  accumulated_metrics['nmi'],
                                                  accumulated_metrics['ARI'],
                                                  accumulated_metrics['fscore'])

                    selected_scores = scores['kmeans']
                    if selected_scores['accuracy'] >= best_acc:
                        best_acc = selected_scores['accuracy']
                        best_nmi = selected_scores['NMI']
                        best_ari = selected_scores['ARI']
                        best_fscore = selected_scores['F-score']
                        self.best_latent_fusion = latent_fusion.copy()
                        self.best_epoch = k + 1
                    metric_print_num = int(config.get('metric_print_num', config['print_num']))
                    metric_now = ((k + 1) == 1) or (metric_print_num > 0 and ((k + 1) % metric_print_num == 0))
                    if metric_now:
                        logger.info(
                            f"Epoch {k + 1} finished: "
                            f"ACC {selected_scores['accuracy']:.4f}, "
                            f"NMI {selected_scores['NMI']:.4f}, "
                            f"ARI {selected_scores['ARI']:.4f}"
                        )

                    self.autoencoder1.train(), self.autoencoder2.train()
                    self.img2txt.train(), self.txt2img.train()

        return best_acc, best_nmi, best_ari, best_fscore


class SharedInferencenBase(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(SharedInferencenBase, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        self.shared_net = nn.Sequential(*layers)

    def forward(self, x):
        return self.shared_net(x)

class SpecificHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SpecificHead, self).__init__()
        self.head = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.head(x)

class PartialSharedCompletion(nn.Module):
    def __init__(self, input_dim, shared_hidden_dims, output_dim, num_views, diveq_cfg=None):
        super(PartialSharedCompletion, self).__init__()
        self.shared_base = SharedInferencenBase(input_dim, shared_hidden_dims)
        self.specific_heads = nn.ModuleList([SpecificHead(shared_hidden_dims[-1], output_dim) for _ in range(num_views)])
        # DIVEQ 量化器（可选）
        self._use_diveq = bool(diveq_cfg.get('enable', True)) if diveq_cfg else True
        if self._use_diveq:
            d = shared_hidden_dims[-1]
            K = int(diveq_cfg.get('K', 64))
            tau = float(diveq_cfg.get('tau', 1.0))
            hard_eval = bool(diveq_cfg.get('hard_eval', True))
            self._diveq = PrototypeConstraintQuantizer(d=d, K=K, tau=tau, hard_eval=hard_eval)
        else:
            self._diveq = None

    def forward(self, x, target_view_idx):
        h = self.shared_base(x)
        if self._diveq is not None and self._use_diveq:
            q = self._diveq(h)
        else:
            q = h
        out = self.specific_heads[target_view_idx](q)
        return out, h

class PCDCUnified(torch.nn.Module):
    # Dual contrastive inference for multi-view
    def __init__(self, config):
        super(PCDCUnified, self).__init__()

        """Constructor.

        Args:
            config: parameters defined in configure.py.
        """
        self._config = config
        self._latent_dim = config['Autoencoder']['arch1'][-1]
        self.view_num = config['view']
        n_clusters = config['training']['class_num']


        for i in range(self.view_num):
            autoencoder = Autoencoder(config['Autoencoder'][f'arch{i + 1}'], config['Autoencoder']['activations'], config['Autoencoder']['batchnorm'])
            self.add_module('autoencoder{}'.format(i), autoencoder)
            dims_view = [self._latent_dim] + self._config['Inference'][f'arch{i + 1}']

        completion_cfg = self._config.get('Completion', self._config['Inference'])
        self.shared_infer_hidden_dims = completion_cfg['shared_hidden']
        self.infer_output_dim = self._latent_dim
        # 读取 DIVEQ 配置
        diveq_cfg = self._config.get('training', {}).get(
            'prototype_constraint',
            self._config.get('training', {}).get('diveq', {'enable': False}),
        )
        self.partial_inferencers = nn.ModuleList([
            PartialSharedCompletion(
                input_dim=self._latent_dim,
                shared_hidden_dims=self.shared_infer_hidden_dims,
                output_dim=self.infer_output_dim,
                num_views=self.view_num,
                diveq_cfg=diveq_cfg
            ) for _ in range(self.view_num)
        ])
        # 轻量 SimSiam 预测头（逐视图独立）
        self.predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self._latent_dim, self._latent_dim),
                nn.BatchNorm1d(self._latent_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self._latent_dim, self._latent_dim)
            ) for _ in range(self.view_num)
        ])
        # 轻量 SimSiam 投影头（逐视图独立）
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self._latent_dim, self._latent_dim),
                nn.BatchNorm1d(self._latent_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self._latent_dim, self._latent_dim)
            ) for _ in range(self.view_num)
        ])
        # 可学习视图融合权重（softmax后作为权重）
        self.view_logits = nn.Parameter(torch.zeros(self.view_num))

        # Best embedding cache for visualization
        self.best_latent_fusion = None
        self.best_scores = None
        self.best_epoch = None

    def _compute_latent_fusion(self, X_list, mask, device):
        self.eval()
        with torch.no_grad():
            # 选择核心视图：可配置（默认关闭，固定为视图0）
            use_core_by_missing = self._config.get('training', {}).get('core_by_missing', False)
            if use_core_by_missing:
                view_avail = mask.sum(dim=0)
                core_view = int(torch.argmax(view_avail).item())
            else:
                core_view = 0
            latent_codes_eval = [torch.zeros(X_list[i].shape[0], self._latent_dim, device=device)
                                 for i in range(self.view_num)]
            for i in range(self.view_num):
                existing_idx_eval = mask[:, i] == 1
                if existing_idx_eval.sum() != 0:
                    latent_codes_eval[i][existing_idx_eval] = getattr(self, f'autoencoder{i}').encoder(
                        X_list[i][existing_idx_eval])
            for i in range(self.view_num):
                if i == core_view:
                    missing_idx_eval = mask[:, i] == 0
                    accumulated = (mask[:, i] == 1).float().to(device)
                    if missing_idx_eval.sum() != 0:
                        for j in range(self.view_num):
                            if j == core_view:
                                continue
                            jhas_idx = missing_idx_eval * (mask[:, j] == 1)
                            accumulated += jhas_idx.float()
                            if jhas_idx.sum() != 0:
                                jhas_latent = latent_codes_eval[j][jhas_idx]
                                inferred_latent, _ = self.partial_inferencers[j](jhas_latent, core_view)
                                latent_codes_eval[i][jhas_idx] += inferred_latent
                    denom = torch.unsqueeze(accumulated.clamp_min(1.0), 1)
                    latent_codes_eval[i] = latent_codes_eval[i] / denom
                else:
                    missing_idx_eval = mask[:, i] == 0
                    if missing_idx_eval.sum() != 0:
                        core_latent = latent_codes_eval[core_view][missing_idx_eval]
                        inferred_latent, _ = self.partial_inferencers[core_view](core_latent, i)
                        latent_codes_eval[i][missing_idx_eval] = inferred_latent

            latent_fusion = torch.cat(latent_codes_eval, dim=1).cpu().numpy()
        self.train()
        return latent_fusion

    def train_multiview(self, config, logger, accumulated_metrics, X_list, Y_list, mask, optimizer, device):

        """Training the model with cove view for clustering

            Args:
              config: parameters which defined in configure.py.
              logger: print the information.
              accumulated_metrics: list of metrics
              X_list: list data of all view
              Y_list: labels
              mask: generate missing data
              optimizer: adam is used in our experiments
              device: to cuda if gpu is used
            Returns:
                            clustering performance: acc, nmi, ari, fscore


        """
        epochs_total = config['training']['epoch']
        batch_size = config['training']['batch_size']

        # 仅使用完整样本
        flag = torch.all(mask == 1, dim=1)
        idx_full = torch.nonzero(flag, as_tuple=True)[0]

        def _rand_choice(idx, size):
            if idx.numel() == 0 or size <= 0:
                return None
            if idx.numel() >= size:
                return idx[torch.randperm(idx.numel(), device=device)[:size]]
            return idx[torch.randint(0, idx.numel(), (size,), device=device)]

        def _mse_per_sample(a, b):
            return ((a - b) ** 2).mean(dim=1)

        def _inf_loss(pred, target, use_conf=False, conf=None, loss_type='mse'):
            if pred is None or target is None:
                return torch.tensor(0.0, device=device)
            if loss_type == 'mse':
                per = _mse_per_sample(pred, target)
            elif loss_type == 'soft_ce':
                per = -(target.clamp_min(1e-8) * F.softmax(pred, dim=1).clamp_min(1e-8).log()).sum(dim=1)
            else:
                raise ValueError(f"Unknown inf_loss: {loss_type}")
            if use_conf and conf is not None:
                per = per * conf.detach().clamp_min(1e-6)
            return per.mean() if per.numel() > 0 else torch.tensor(0.0, device=device)

        use_conf = config['training'].get('inf_conf_weight', False)
        conf_thresh = float(config['training'].get('conf_thresh', 0.7))

        best_acc, best_nmi, best_ari, best_fscore = 0, 0, 0, 0
        select_best_by_label = bool(config['training'].get('select_best_by_label', False))
        if 'fscore' not in accumulated_metrics:
            accumulated_metrics['fscore'] = []
        start_completion = int(config['training'].get('start_completion', config['training'].get('start_inference', 0)))

        for k in range(epochs_total):
            loss_all, rec, dul, icl = 0, 0, 0, 0

            b_full = batch_size if idx_full.numel() > 0 else 0

            steps = 1
            if b_full > 0:
                steps = max(steps, math.ceil(idx_full.numel() / b_full))
            for _ in range(steps):
                # 采样完整样本
                idx_f = _rand_choice(idx_full, b_full)
                batch_full = None
                if idx_f is not None:
                    batch_full = [X_list[i][idx_f].to(device).float() for i in range(self.view_num)]

                # Within-view Reconstruction Loss（仅完整样本）
                reconstruction_loss = 0.0
                if batch_full is not None:
                    for i in range(self.view_num):
                        autoencoder = getattr(self, f'autoencoder{i}')
                        z_recon = autoencoder.encoder(batch_full[i])
                        reconstruction_loss += F.mse_loss(autoencoder.decoder(z_recon), batch_full[i])
                    reconstruction_loss = reconstruction_loss / max(1, self.view_num)

                # 可学习视图权重融合：构建全局锚 H，并对每视图施加锚一致性-----------------------------------------------------------------------
                #w = torch.softmax(self.view_logits, dim=0).to(latent_view_z[0].device)
                #H = sum(latent_view_z[i] * w[i] for i in range(self.view_num))
                #loss_anchor = 0.0
                #for i in range(self.view_num):
                #    loss_anchor += F.mse_loss(latent_view_z[i], H)
                #loss_anchor /= self.view_num------------------------------------------------------------------------------------------------

                # Instance-level Contrastive + SimSiam 对齐（仅完整样本）
                icl_inst = torch.tensor(0.0, device=device)
                simsiam_loss = torch.tensor(0.0, device=device)
                pair_count = 0
                latent_view_z = None
                if batch_full is not None:
                    latent_view_z = []
                    for i in range(self.view_num):
                        autoencoder = getattr(self, f'autoencoder{i}')
                        latent_view_z.append(autoencoder.encoder(batch_full[i]))
                    z_proj = [self.projectors[i](latent_view_z[i]) for i in range(self.view_num)]
                    for i in range(self.view_num):
                        for j in range(i + 1, self.view_num):
                            icl_inst += maximize_mutual_information_loss(latent_view_z[i], latent_view_z[j],
                                                                  config['training']['alpha'])
                            p_i = self.predictors[i](z_proj[i])
                            p_j = self.predictors[j](z_proj[j])
                            simsiam_loss += siamese_similarity_contrastive_loss(p_i, p_j,
                                                          z_proj[i].detach(),
                                                          z_proj[j].detach())
                            pair_count += 1
                    if pair_count > 0:
                        icl_inst /= pair_count
                        simsiam_loss /= pair_count
                icl_loss = icl_inst * config['training']['lambda2'] + simsiam_loss * config['training']['lambda3']

                # ----------------修改部分3--------------------
                # Redundancy reduction on aligned embeddings, adaptive to missing_rate
                #with torch.no_grad():
                #    mask_ratio = float(config.get('missing_rate', 0.0))
                #stacked_latent = torch.cat(latent_view_z, dim=0)
                #cov_loss = covariance_penalty(stacked_latent)
                #align_w = 1.0 - mask_ratio
                #cov_w = config['training'].get('lambda_cov', 1.0) * (0.5 + 0.5 * mask_ratio)
                #icl_loss = align_w * icl_loss + cov_w * cov_loss
                # ----------------修改部分3--------------------

                # Dual-inference（完整样本：直推 + 循环一致性）
                dualinference_loss = torch.tensor(0.0, device=device)
                direct_loss = torch.tensor(0.0, device=device)
                cycle_loss = torch.tensor(0.0, device=device)
                count_direct = 0
                count_cycle = 0
                inf_loss_type = config['training'].get('inf_loss', 'mse')
                do_sharpen = config['training'].get('inf_sharpen', False)

                if latent_view_z is not None:
                    for i in range(self.view_num):
                        for j in range(self.view_num):
                            if i == j:
                                continue
                            infer_ij, _ = self.partial_inferencers[i](latent_view_z[i], j)
                            # 直推：i -> j 对齐 z_j
                            target_direct = latent_view_z[j]
                            if do_sharpen:
                                target_direct = target_l2(target_direct)
                            if use_conf:
                                conf = F.softmax(infer_ij, dim=1).max(dim=1).values
                                valid = conf > conf_thresh
                                if valid.sum() != 0:
                                    direct_loss += _inf_loss(
                                        infer_ij[valid], target_direct.detach()[valid], True, conf[valid], inf_loss_type
                                    )
                                    count_direct += 1
                            else:
                                direct_loss += _inf_loss(infer_ij, target_direct.detach(), False, None, inf_loss_type)
                                count_direct += 1

                            # 循环：i -> j -> i 对齐 z_i
                            infer_ji, _ = self.partial_inferencers[j](infer_ij, i)
                            target_cycle = latent_view_z[i]
                            if do_sharpen:
                                target_cycle = target_l2(target_cycle)
                            if use_conf:
                                conf = F.softmax(infer_ij, dim=1).max(dim=1).values
                                valid = conf > conf_thresh
                                if valid.sum() != 0:
                                    cycle_loss += _inf_loss(
                                        infer_ji[valid], target_cycle.detach()[valid], True, conf[valid], inf_loss_type
                                    )
                                    count_cycle += 1
                            else:
                                cycle_loss += _inf_loss(infer_ji, target_cycle.detach(), False, None, inf_loss_type)
                                count_cycle += 1

                if count_direct > 0:
                    direct_loss = direct_loss / count_direct
                if count_cycle > 0:
                    cycle_loss = cycle_loss / count_cycle

                dualinference_loss = direct_loss + cycle_loss

                all_loss = reconstruction_loss * config['training']['lambda1']
                #all_loss += loss_anchor * config['training'].get('lambda_anchor', 1.0)-----------------------------------------------------------------------
                all_loss += icl_loss
                if k >= start_completion:
                    all_loss += config['training']['lambda4'] * dualinference_loss

                optimizer.zero_grad()
                all_loss.backward()
                optimizer.step()

                loss_all += all_loss.item()


            eval_now = ((k + 1) == 1) or (((k + 1) % config['print_num']) == 0) or ((k + 1) == epochs_total)

            if eval_now:
                with torch.no_grad():
                    latent_fusion = self._compute_latent_fusion(X_list, mask, device)

                    scores = clustering.get_score(
                        [latent_fusion], Y_list,
                        accumulated_metrics['acc'], accumulated_metrics['nmi'],
                        accumulated_metrics['ARI'], accumulated_metrics['fscore']
                    )

                    selected_scores = scores['kmeans']
                    current_acc = selected_scores['accuracy']
                    current_nmi = selected_scores['NMI']
                    current_ari = selected_scores['ARI']
                    current_fscore = selected_scores['F-score']

                    if select_best_by_label:
                        # Legacy behavior (not recommended for fair reporting):
                        # select checkpoint by supervised metric.
                        if current_acc >= best_acc:
                            best_acc = current_acc
                            best_nmi = current_nmi
                            best_ari = current_ari
                            best_fscore = current_fscore
                            self.best_latent_fusion = latent_fusion.copy()
                            self.best_scores = selected_scores
                            self.best_epoch = k + 1
                    else:
                        # Fair default: report the latest evaluated checkpoint.
                        best_acc = current_acc
                        best_nmi = current_nmi
                        best_ari = current_ari
                        best_fscore = current_fscore
                        self.best_latent_fusion = latent_fusion.copy()
                        self.best_scores = selected_scores
                        self.best_epoch = k + 1
                    metric_print_num = int(config.get('metric_print_num', config['print_num']))
                    metric_now = ((k + 1) == 1) or (metric_print_num > 0 and ((k + 1) % metric_print_num == 0))
                    if metric_now:
                        logger.info(
                            f"Epoch {k + 1} finished: "
                            f"ACC {current_acc:.4f}, "
                            f"NMI {current_nmi:.4f}, "
                            f"ARI {current_ari:.4f}"
                        )
        return best_acc, best_nmi, best_ari, best_fscore



def get_model_class(view_num: int):
    if int(view_num) == 2:
        return PCDC
    return PCDCUnified


def build_model(config: dict, mode: Literal["auto", "two", "multi"] = "auto"):
    if mode == "two":
        return PCDC(config)
    if mode == "multi":
        return PCDCUnified(config)

    view_num = int(config.get("view", 0))
    model_cls = get_model_class(view_num)
    return model_cls(config)


# Backward-compatible aliases.
Inference = CompletionModule
DIVEQQuantizer = PrototypeConstraintQuantizer
model = PCDC
model_multi_view = PCDCUnified