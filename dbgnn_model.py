import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
import numpy as np
import math
from typing import Optional, Tuple


class BayesianLinear(nn.Module):
    """贝叶斯线性层，使用变分推理"""
    
    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        
        # 权重参数
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_logvar = nn.Parameter(torch.randn(out_features, in_features) * 0.1 - 5)
        
        # 偏置参数
        self.bias_mu = nn.Parameter(torch.randn(out_features) * 0.1)
        self.bias_logvar = nn.Parameter(torch.randn(out_features) * 0.1 - 5)
        
    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if sample:
            # 重参数化技巧
            weight_std = torch.exp(0.5 * self.weight_logvar)
            bias_std = torch.exp(0.5 * self.bias_logvar)
            
            weight_eps = torch.randn_like(weight_std)
            bias_eps = torch.randn_like(bias_std)
            
            weight = self.weight_mu + weight_std * weight_eps
            bias = self.bias_mu + bias_std * bias_eps
        else:
            weight = self.weight_mu
            bias = self.bias_mu
            
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """计算KL散度损失"""
        # 权重的KL散度
        weight_kl = -0.5 * torch.sum(
            1 + self.weight_logvar - self.weight_mu.pow(2) / (self.prior_std ** 2) 
            - torch.exp(self.weight_logvar) / (self.prior_std ** 2)
        )
        
        # 偏置的KL散度
        bias_kl = -0.5 * torch.sum(
            1 + self.bias_logvar - self.bias_mu.pow(2) / (self.prior_std ** 2)
            - torch.exp(self.bias_logvar) / (self.prior_std ** 2)
        )
        
        return weight_kl + bias_kl


class DBGNN(nn.Module):
    """Deep Bayesian Graph Neural Network for Stock Prediction"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        num_layers: int = 3,
        dropout: float = 0.1,
        prior_std: float = 1.0
    ):
        super(DBGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 图卷积层
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # 贝叶斯全连接层
        self.bayesian_layers = nn.ModuleList()
        self.bayesian_layers.append(BayesianLinear(hidden_dim * 2, hidden_dim, prior_std))
        self.bayesian_layers.append(BayesianLinear(hidden_dim, hidden_dim // 2, prior_std))
        self.bayesian_layers.append(BayesianLinear(hidden_dim // 2, output_dim, prior_std))
        
        # Dropout层
        self.dropout_layer = nn.Dropout(dropout)
        
        # 批归一化
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        batch: Optional[torch.Tensor] = None,
        sample: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, input_dim]
            edge_index: 边索引 [2, num_edges]
            batch: 批次索引 [num_nodes]
            sample: 是否从后验分布采样
            
        Returns:
            预测值和不确定性估计
        """
        # 图卷积层
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout_layer(x)
        
        # 图级别池化
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # 使用均值和最大值池化
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        # 贝叶斯全连接层
        for i, layer in enumerate(self.bayesian_layers[:-1]):
            x = layer(x, sample=sample)
            x = F.relu(x)
            x = self.dropout_layer(x)
        
        # 输出层
        output = self.bayesian_layers[-1](x, sample=sample)
        
        if sample:
            # 计算不确定性（通过多次采样）
            uncertainty = self._compute_uncertainty(x)
        else:
            uncertainty = torch.zeros_like(output)
        
        return output, uncertainty
    
    def _compute_uncertainty(self, x: torch.Tensor, num_samples: int = 10) -> torch.Tensor:
        """通过多次采样计算不确定性"""
        outputs = []
        for _ in range(num_samples):
            out = self.bayesian_layers[-1](x, sample=True)
            outputs.append(out)
        
        outputs = torch.stack(outputs)
        uncertainty = torch.std(outputs, dim=0)
        return uncertainty
    
    def kl_divergence(self) -> torch.Tensor:
        """计算所有贝叶斯层的KL散度"""
        kl_div = 0
        for layer in self.bayesian_layers:
            kl_div += layer.kl_divergence()
        return kl_div
    
    def predict_with_uncertainty(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        batch: Optional[torch.Tensor] = None,
        num_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        预测并返回不确定性估计
        
        Returns:
            mean_prediction: 平均预测
            aleatoric_uncertainty: 偶然不确定性
            epistemic_uncertainty: 认知不确定性
        """
        self.eval()
        with torch.no_grad():
            predictions = []
            uncertainties = []
            
            for _ in range(num_samples):
                pred, unc = self.forward(x, edge_index, batch, sample=True)
                predictions.append(pred)
                uncertainties.append(unc)
            
            predictions = torch.stack(predictions)
            uncertainties = torch.stack(uncertainties)
            
            # 平均预测
            mean_prediction = torch.mean(predictions, dim=0)
            
            # 认知不确定性（模型不确定性）
            epistemic_uncertainty = torch.std(predictions, dim=0)
            
            # 偶然不确定性（数据噪声）
            aleatoric_uncertainty = torch.mean(uncertainties, dim=0)
            
        return mean_prediction, aleatoric_uncertainty, epistemic_uncertainty


class DBGNNLoss(nn.Module):
    """DBGNN的损失函数，包含重构损失和KL散度"""
    
    def __init__(self, kl_weight: float = 1e-4):
        super(DBGNNLoss, self).__init__()
        self.kl_weight = kl_weight
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self, 
        prediction: torch.Tensor, 
        target: torch.Tensor, 
        kl_divergence: torch.Tensor,
        num_samples: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算总损失
        
        Returns:
            total_loss, reconstruction_loss, kl_loss
        """
        # 重构损失
        reconstruction_loss = self.mse_loss(prediction, target)
        
        # KL散度损失
        kl_loss = kl_divergence / num_samples
        
        # 总损失
        total_loss = reconstruction_loss + self.kl_weight * kl_loss
        
        return total_loss, reconstruction_loss, kl_loss