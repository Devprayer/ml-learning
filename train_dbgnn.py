import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import List, Tuple, Dict
import os
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from dbgnn_model import DBGNN, DBGNNLoss
from stock_data_generator import StockDataGenerator


class DBGNNTrainer:
    """DBGNN模型训练器"""
    
    def __init__(self, 
                 model: DBGNN,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 log_dir: str = 'logs',
                 model_save_dir: str = 'models'):
        
        self.model = model.to(device)
        self.device = device
        self.log_dir = log_dir
        self.model_save_dir = model_save_dir
        
        # 创建目录
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_save_dir, exist_ok=True)
        
        # 初始化TensorBoard
        self.writer = SummaryWriter(log_dir)
        
        # 训练历史
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'train_mse': [],
            'train_kl': [],
            'val_loss': [],
            'val_mse': [],
            'val_kl': [],
            'val_r2': []
        }
        
    def prepare_data_loaders(self, 
                           graph_data_list: List,
                           train_ratio: float = 0.7,
                           val_ratio: float = 0.15,
                           batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """准备训练、验证和测试数据加载器"""
        
        # 数据划分
        n_total = len(graph_data_list)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_data = graph_data_list[:n_train]
        val_data = graph_data_list[n_train:n_train + n_val]
        test_data = graph_data_list[n_train + n_val:]
        
        # 创建数据加载器
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
        print(f"训练集大小: {len(train_data)}")
        print(f"验证集大小: {len(val_data)}")
        print(f"测试集大小: {len(test_data)}")
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, 
                   train_loader: DataLoader, 
                   optimizer: optim.Optimizer, 
                   criterion: DBGNNLoss) -> Dict[str, float]:
        """训练一个epoch"""
        
        self.model.train()
        epoch_loss = 0
        epoch_mse = 0
        epoch_kl = 0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            batch = batch.to(self.device)
            optimizer.zero_grad()
            
            # 前向传播
            predictions, uncertainties = self.model(batch.x, batch.edge_index, batch.batch)
            
            # 计算损失
            kl_div = self.model.kl_divergence()
            total_loss, mse_loss, kl_loss = criterion(
                predictions, batch.y, kl_div, num_samples=len(batch.y)
            )
            
            # 反向传播
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 累计损失
            epoch_loss += total_loss.item()
            epoch_mse += mse_loss.item()
            epoch_kl += kl_loss.item()
            num_batches += 1
        
        return {
            'loss': epoch_loss / num_batches,
            'mse': epoch_mse / num_batches,
            'kl': epoch_kl / num_batches
        }
    
    def validate_epoch(self, 
                      val_loader: DataLoader, 
                      criterion: DBGNNLoss) -> Dict[str, float]:
        """验证一个epoch"""
        
        self.model.eval()
        epoch_loss = 0
        epoch_mse = 0
        epoch_kl = 0
        all_predictions = []
        all_targets = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = batch.to(self.device)
                
                # 前向传播
                predictions, uncertainties = self.model(batch.x, batch.edge_index, batch.batch)
                
                # 计算损失
                kl_div = self.model.kl_divergence()
                total_loss, mse_loss, kl_loss = criterion(
                    predictions, batch.y, kl_div, num_samples=len(batch.y)
                )
                
                # 累计损失和预测
                epoch_loss += total_loss.item()
                epoch_mse += mse_loss.item()
                epoch_kl += kl_loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch.y.cpu().numpy())
                num_batches += 1
        
        # 计算R²分数
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        r2 = r2_score(all_targets, all_predictions)
        
        return {
            'loss': epoch_loss / num_batches,
            'mse': epoch_mse / num_batches,
            'kl': epoch_kl / num_batches,
            'r2': r2
        }
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 100,
              learning_rate: float = 0.001,
              weight_decay: float = 1e-5,
              kl_weight: float = 1e-4,
              patience: int = 10,
              min_delta: float = 1e-5) -> Dict:
        """训练模型"""
        
        # 初始化优化器和损失函数
        optimizer = optim.Adam(self.model.parameters(), 
                             lr=learning_rate, 
                             weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        criterion = DBGNNLoss(kl_weight=kl_weight)
        
        # 早停机制
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0
        
        print(f"开始训练，总共 {epochs} 个epoch")
        print(f"设备: {self.device}")
        print(f"学习率: {learning_rate}")
        print(f"KL权重: {kl_weight}")
        
        for epoch in range(epochs):
            # 训练
            train_metrics = self.train_epoch(train_loader, optimizer, criterion)
            
            # 验证
            val_metrics = self.validate_epoch(val_loader, criterion)
            
            # 学习率调度
            scheduler.step(val_metrics['loss'])
            
            # 记录历史
            self.train_history['epoch'].append(epoch)
            self.train_history['train_loss'].append(train_metrics['loss'])
            self.train_history['train_mse'].append(train_metrics['mse'])
            self.train_history['train_kl'].append(train_metrics['kl'])
            self.train_history['val_loss'].append(val_metrics['loss'])
            self.train_history['val_mse'].append(val_metrics['mse'])
            self.train_history['val_kl'].append(val_metrics['kl'])
            self.train_history['val_r2'].append(val_metrics['r2'])
            
            # TensorBoard记录
            self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/Validation', val_metrics['loss'], epoch)
            self.writer.add_scalar('MSE/Train', train_metrics['mse'], epoch)
            self.writer.add_scalar('MSE/Validation', val_metrics['mse'], epoch)
            self.writer.add_scalar('KL/Train', train_metrics['kl'], epoch)
            self.writer.add_scalar('KL/Validation', val_metrics['kl'], epoch)
            self.writer.add_scalar('R2/Validation', val_metrics['r2'], epoch)
            self.writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            # 打印进度
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:3d}/{epochs}")
                print(f"  Train - Loss: {train_metrics['loss']:.6f}, MSE: {train_metrics['mse']:.6f}, KL: {train_metrics['kl']:.6f}")
                print(f"  Val   - Loss: {val_metrics['loss']:.6f}, MSE: {val_metrics['mse']:.6f}, KL: {val_metrics['kl']:.6f}, R²: {val_metrics['r2']:.4f}")
                print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 早停检查
            if val_metrics['loss'] < best_val_loss - min_delta:
                best_val_loss = val_metrics['loss']
                best_epoch = epoch
                patience_counter = 0
                
                # 保存最佳模型
                self.save_model(f'best_model_epoch_{epoch}.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
                break
        
        self.writer.close()
        
        # 保存训练历史
        with open(os.path.join(self.log_dir, 'training_history.pkl'), 'wb') as f:
            pickle.dump(self.train_history, f)
        
        return self.train_history
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """评估模型"""
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_uncertainties = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                batch = batch.to(self.device)
                
                # 使用不确定性预测
                mean_pred, aleatoric_unc, epistemic_unc = self.model.predict_with_uncertainty(
                    batch.x, batch.edge_index, batch.batch, num_samples=50
                )
                
                all_predictions.extend(mean_pred.cpu().numpy())
                all_targets.extend(batch.y.cpu().numpy())
                all_uncertainties.extend((aleatoric_unc + epistemic_unc).cpu().numpy())
        
        # 计算评估指标
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_uncertainties = np.array(all_uncertainties)
        
        mse = mean_squared_error(all_targets, all_predictions)
        mae = mean_absolute_error(all_targets, all_predictions)
        r2 = r2_score(all_targets, all_predictions)
        
        # 计算方向准确率
        pred_direction = np.sign(all_predictions.flatten())
        true_direction = np.sign(all_targets.flatten())
        direction_accuracy = np.mean(pred_direction == true_direction)
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'direction_accuracy': direction_accuracy,
            'mean_uncertainty': np.mean(all_uncertainties)
        }
        
        print("=== 测试集评估结果 ===")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.6f}")
        
        return metrics, all_predictions, all_targets, all_uncertainties
    
    def save_model(self, filename: str):
        """保存模型"""
        filepath = os.path.join(self.model_save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_dim': self.model.input_dim,
                'hidden_dim': self.model.hidden_dim,
                'output_dim': self.model.output_dim,
                'num_layers': self.model.num_layers,
                'dropout': self.model.dropout
            }
        }, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型已从 {filepath} 加载")
    
    def plot_training_history(self):
        """绘制训练历史"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(self.train_history['epoch'], self.train_history['train_loss'], 
                       label='Train Loss', color='blue')
        axes[0, 0].plot(self.train_history['epoch'], self.train_history['val_loss'], 
                       label='Val Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MSE曲线
        axes[0, 1].plot(self.train_history['epoch'], self.train_history['train_mse'], 
                       label='Train MSE', color='blue')
        axes[0, 1].plot(self.train_history['epoch'], self.train_history['val_mse'], 
                       label='Val MSE', color='red')
        axes[0, 1].set_title('Mean Squared Error')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MSE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # KL散度曲线
        axes[1, 0].plot(self.train_history['epoch'], self.train_history['train_kl'], 
                       label='Train KL', color='blue')
        axes[1, 0].plot(self.train_history['epoch'], self.train_history['val_kl'], 
                       label='Val KL', color='red')
        axes[1, 0].set_title('KL Divergence')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('KL Divergence')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # R²分数
        axes[1, 1].plot(self.train_history['epoch'], self.train_history['val_r2'], 
                       label='Val R²', color='green')
        axes[1, 1].set_title('R² Score')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('R² Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_history.png'), dpi=300)
        plt.show()


def main():
    """主训练函数"""
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("=== DBGNN股票预测模型训练 ===")
    
    # 生成数据
    print("生成模拟股票数据...")
    generator = StockDataGenerator(num_stocks=30, time_periods=500, seed=42)
    price_data = generator.generate_stock_prices()
    features_data = generator.calculate_technical_indicators(price_data)
    graph_data_list = generator.prepare_time_series_data(price_data, features_data)
    
    print(f"生成了 {len(graph_data_list)} 个图数据样本")
    print(f"节点特征维度: {graph_data_list[0].x.shape[1]}")
    
    # 初始化模型
    input_dim = graph_data_list[0].x.shape[1]
    model = DBGNN(
        input_dim=input_dim,
        hidden_dim=64,
        output_dim=1,
        num_layers=3,
        dropout=0.1,
        prior_std=1.0
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 初始化训练器
    trainer = DBGNNTrainer(model)
    
    # 准备数据加载器
    train_loader, val_loader, test_loader = trainer.prepare_data_loaders(
        graph_data_list, batch_size=16
    )
    
    # 训练模型
    print("\n开始训练...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=200,
        learning_rate=0.001,
        kl_weight=1e-4,
        patience=20
    )
    
    # 绘制训练历史
    trainer.plot_training_history()
    
    # 评估模型
    print("\n评估模型...")
    metrics, predictions, targets, uncertainties = trainer.evaluate(test_loader)
    
    # 可视化预测结果
    plt.figure(figsize=(15, 5))
    
    # 预测vs真实值
    plt.subplot(1, 3, 1)
    plt.scatter(targets, predictions, alpha=0.6)
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'Predictions vs True Values (R²={metrics["r2"]:.3f})')
    plt.grid(True)
    
    # 残差图
    plt.subplot(1, 3, 2)
    residuals = predictions.flatten() - targets.flatten()
    plt.scatter(predictions.flatten(), residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predictions')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True)
    
    # 不确定性分布
    plt.subplot(1, 3, 3)
    plt.hist(uncertainties.flatten(), bins=30, alpha=0.7)
    plt.xlabel('Uncertainty')
    plt.ylabel('Frequency')
    plt.title('Uncertainty Distribution')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(trainer.log_dir, 'evaluation_results.png'), dpi=300)
    plt.show()
    
    print("\n训练完成！")


if __name__ == "__main__":
    main()