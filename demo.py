#!/usr/bin/env python3
"""
DBGNN股票预测模型演示脚本

这个脚本演示了如何：
1. 生成模拟股票数据
2. 训练DBGNN模型
3. 进行预测和不确定性估计
4. 可视化结果
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from dbgnn_model import DBGNN, DBGNNLoss
from stock_data_generator import StockDataGenerator
from train_dbgnn import DBGNNTrainer


def quick_demo():
    """快速演示：生成数据并展示基本功能"""
    
    print("🚀 DBGNN股票预测模型演示")
    print("=" * 50)
    
    # 1. 数据生成
    print("\n📊 第一步：生成模拟股票数据")
    generator = StockDataGenerator(num_stocks=20, time_periods=200, seed=42)
    
    # 生成价格数据
    price_data = generator.generate_stock_prices(
        initial_price=100.0,
        volatility=0.02,
        drift=0.0005
    )
    
    # 计算技术指标
    features_data = generator.calculate_technical_indicators(price_data)
    
    # 展示数据统计
    generator.get_data_statistics(price_data, features_data)
    
    # 可视化样本股票
    print("\n📈 股票价格走势图")
    generator.plot_sample_stocks(price_data, num_samples=5)
    
    # 可视化股票关系网络
    print("\n🕸️ 股票相关性网络图")
    generator.visualize_stock_network(correlation_threshold=0.25)
    
    return generator, price_data, features_data


def train_demo_model(generator, price_data, features_data):
    """训练演示模型"""
    
    print("\n🤖 第二步：训练DBGNN模型")
    
    # 准备图数据
    graph_data_list = generator.prepare_time_series_data(
        price_data, features_data, sequence_length=10, prediction_horizon=1
    )
    
    print(f"生成了 {len(graph_data_list)} 个图数据样本")
    print(f"节点特征维度: {graph_data_list[0].x.shape[1]}")
    print(f"边数量: {graph_data_list[0].edge_index.shape[1]}")
    
    # 初始化模型
    input_dim = graph_data_list[0].x.shape[1]
    model = DBGNN(
        input_dim=input_dim,
        hidden_dim=32,  # 减少隐藏层维度加快训练
        output_dim=1,
        num_layers=2,   # 减少层数
        dropout=0.1,
        prior_std=1.0
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 初始化训练器
    trainer = DBGNNTrainer(model, log_dir='demo_logs', model_save_dir='demo_models')
    
    # 准备数据加载器
    train_loader, val_loader, test_loader = trainer.prepare_data_loaders(
        graph_data_list, batch_size=8
    )
    
    # 快速训练（减少epoch数用于演示）
    print("\n🏃‍♂️ 开始快速训练（演示模式）...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,      # 减少epoch数
        learning_rate=0.01,
        kl_weight=1e-3,
        patience=10
    )
    
    return trainer, test_loader, history


def analyze_predictions(trainer, test_loader):
    """分析预测结果"""
    
    print("\n📊 第三步：分析预测结果")
    
    # 评估模型
    metrics, predictions, targets, uncertainties = trainer.evaluate(test_loader)
    
    # 创建详细的可视化
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 预测vs真实值散点图
    plt.subplot(3, 3, 1)
    plt.scatter(targets, predictions, alpha=0.6, c=uncertainties, cmap='viridis')
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
    plt.xlabel('True Return Rate')
    plt.ylabel('Predicted Return Rate')
    plt.title(f'Predictions vs True Values\n(R²={metrics["r2"]:.3f})')
    plt.colorbar(label='Uncertainty')
    plt.grid(True, alpha=0.3)
    
    # 2. 残差分析
    plt.subplot(3, 3, 2)
    residuals = predictions.flatten() - targets.flatten()
    plt.scatter(predictions.flatten(), residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predictions')
    plt.ylabel('Residuals')
    plt.title('Residual Analysis')
    plt.grid(True, alpha=0.3)
    
    # 3. 不确定性分布
    plt.subplot(3, 3, 3)
    plt.hist(uncertainties.flatten(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Uncertainty')
    plt.ylabel('Frequency')
    plt.title('Prediction Uncertainty Distribution')
    plt.grid(True, alpha=0.3)
    
    # 4. 预测准确性分箱分析
    plt.subplot(3, 3, 4)
    uncertainty_bins = np.percentile(uncertainties, [0, 25, 50, 75, 100])
    bin_labels = ['Low', 'Medium-Low', 'Medium-High', 'High']
    bin_accuracies = []
    
    for i in range(len(uncertainty_bins)-1):
        mask = (uncertainties >= uncertainty_bins[i]) & (uncertainties < uncertainty_bins[i+1])
        if i == len(uncertainty_bins)-2:  # 最后一个bin包含最大值
            mask = (uncertainties >= uncertainty_bins[i]) & (uncertainties <= uncertainty_bins[i+1])
        
        if np.sum(mask) > 0:
            bin_predictions = predictions[mask]
            bin_targets = targets[mask]
            bin_mse = np.mean((bin_predictions - bin_targets)**2)
            bin_accuracies.append(bin_mse)
        else:
            bin_accuracies.append(0)
    
    plt.bar(bin_labels, bin_accuracies, alpha=0.7, color='coral')
    plt.xlabel('Uncertainty Level')
    plt.ylabel('Mean Squared Error')
    plt.title('Prediction Error by Uncertainty Level')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 5. 方向准确性分析
    plt.subplot(3, 3, 5)
    pred_direction = np.sign(predictions.flatten())
    true_direction = np.sign(targets.flatten())
    
    # 创建混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_direction, pred_direction, labels=[-1, 1])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
    plt.xlabel('Predicted Direction')
    plt.ylabel('True Direction')
    plt.title(f'Direction Accuracy: {metrics["direction_accuracy"]:.3f}')
    
    # 6. 时间序列预测示例
    plt.subplot(3, 3, 6)
    # 选择前50个样本进行可视化
    sample_size = min(50, len(predictions))
    time_steps = range(sample_size)
    plt.plot(time_steps, targets[:sample_size], 'b-', label='True', linewidth=2)
    plt.plot(time_steps, predictions[:sample_size], 'r--', label='Predicted', linewidth=2)
    plt.fill_between(time_steps, 
                     predictions[:sample_size].flatten() - uncertainties[:sample_size].flatten(),
                     predictions[:sample_size].flatten() + uncertainties[:sample_size].flatten(),
                     alpha=0.3, color='red', label='Uncertainty')
    plt.xlabel('Time Steps')
    plt.ylabel('Return Rate')
    plt.title('Time Series Prediction Sample')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. 特征重要性分析（基于梯度）
    plt.subplot(3, 3, 7)
    # 简化的特征重要性可视化
    feature_names = ['Sector', 'Price_Mean', 'Price_Std', 'Return', 'MA_Short', 'MA_Long', 'RSI', 'Volatility', 'Price_Position']
    # 模拟特征重要性（实际应用中需要计算真实的特征重要性）
    importance_scores = np.random.rand(len(feature_names))
    importance_scores = importance_scores / importance_scores.sum()
    
    plt.barh(feature_names, importance_scores, alpha=0.7, color='lightgreen')
    plt.xlabel('Relative Importance')
    plt.title('Feature Importance (Simulated)')
    plt.grid(True, alpha=0.3)
    
    # 8. 损失函数组件分析
    plt.subplot(3, 3, 8)
    history = trainer.train_history
    epochs = history['epoch']
    plt.plot(epochs, history['train_mse'], label='Train MSE', color='blue')
    plt.plot(epochs, history['val_mse'], label='Val MSE', color='red')
    plt.plot(epochs, np.array(history['train_kl'])*1000, label='Train KL×1000', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Components')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 9. 不确定性校准图
    plt.subplot(3, 3, 9)
    # 将不确定性分为10个区间
    n_bins = 10
    uncertainty_percentiles = np.linspace(0, 100, n_bins+1)
    bin_boundaries = np.percentile(uncertainties, uncertainty_percentiles)
    
    bin_centers = []
    bin_errors = []
    
    for i in range(n_bins):
        if i == n_bins-1:
            mask = (uncertainties >= bin_boundaries[i]) & (uncertainties <= bin_boundaries[i+1])
        else:
            mask = (uncertainties >= bin_boundaries[i]) & (uncertainties < bin_boundaries[i+1])
        
        if np.sum(mask) > 0:
            bin_uncertainty = uncertainties[mask].mean()
            bin_error = np.abs(predictions[mask] - targets[mask]).mean()
            bin_centers.append(bin_uncertainty)
            bin_errors.append(bin_error)
    
    plt.scatter(bin_centers, bin_errors, alpha=0.7, s=50, color='purple')
    if bin_centers and bin_errors:
        plt.plot([min(bin_centers), max(bin_centers)], 
                [min(bin_centers), max(bin_centers)], 'r--', label='Perfect Calibration')
    plt.xlabel('Predicted Uncertainty')
    plt.ylabel('Actual Error')
    plt.title('Uncertainty Calibration')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('demo_results_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return metrics, predictions, targets, uncertainties


def create_interactive_visualization(generator, price_data, predictions, targets, uncertainties):
    """创建交互式可视化"""
    
    print("\n🎨 第四步：创建交互式可视化")
    
    # 创建Plotly交互式图表
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Stock Price Evolution', 'Prediction vs Reality', 
                       'Uncertainty Analysis', 'Performance Metrics'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "table"}]]
    )
    
    # 1. 股票价格演变
    sample_stocks = generator.stock_symbols[:5]
    for stock in sample_stocks:
        fig.add_trace(
            go.Scatter(x=price_data.index, y=price_data[stock], 
                      mode='lines', name=stock, opacity=0.7),
            row=1, col=1
        )
    
    # 2. 预测vs真实值
    fig.add_trace(
        go.Scatter(x=targets.flatten(), y=predictions.flatten(),
                  mode='markers', name='Predictions',
                  marker=dict(color=uncertainties.flatten(), 
                            colorscale='Viridis', showscale=True,
                            colorbar=dict(title="Uncertainty")),
                  text=[f'Uncertainty: {u:.4f}' for u in uncertainties.flatten()]),
        row=1, col=2
    )
    
    # 添加理想线
    min_val, max_val = targets.min(), targets.max()
    fig.add_trace(
        go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                  mode='lines', name='Perfect Prediction', 
                  line=dict(dash='dash', color='red')),
        row=1, col=2
    )
    
    # 3. 不确定性分析
    fig.add_trace(
        go.Histogram(x=uncertainties.flatten(), name='Uncertainty Distribution',
                    opacity=0.7, nbinsx=30),
        row=2, col=1
    )
    
    # 4. 性能指标表格
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    
    # 计算方向准确率
    pred_direction = np.sign(predictions.flatten())
    true_direction = np.sign(targets.flatten())
    direction_accuracy = np.mean(pred_direction == true_direction)
    
    metrics_table = go.Table(
        header=dict(values=['Metric', 'Value'],
                   fill_color='lightblue'),
        cells=dict(values=[['MSE', 'MAE', 'R²', 'Direction Accuracy', 'Mean Uncertainty'],
                          [f'{mse:.6f}', f'{mae:.6f}', f'{r2:.4f}', 
                           f'{direction_accuracy:.4f}', f'{np.mean(uncertainties):.6f}']],
                  fill_color='white')
    )
    
    fig.add_trace(metrics_table, row=2, col=2)
    
    # 更新布局
    fig.update_layout(
        title_text="DBGNN Stock Prediction Model - Interactive Dashboard",
        title_x=0.5,
        height=800,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_xaxes(title_text="True Values", row=1, col=2)
    fig.update_yaxes(title_text="Predictions", row=1, col=2)
    fig.update_xaxes(title_text="Uncertainty", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)
    
    # 保存为HTML文件
    fig.write_html("dbgnn_interactive_dashboard.html")
    print("📱 交互式Dashboard已保存为 'dbgnn_interactive_dashboard.html'")
    
    # 显示图表
    fig.show()


def generate_prediction_report(metrics, predictions, targets, uncertainties):
    """生成预测报告"""
    
    print("\n📋 第五步：生成预测报告")
    
    report = f"""
    ========================================
           DBGNN 股票预测模型报告
    ========================================
    
    模型性能指标:
    ----------------------------------------
    • 均方误差 (MSE):          {metrics['mse']:.6f}
    • 平均绝对误差 (MAE):      {metrics['mae']:.6f}
    • 决定系数 (R²):           {metrics['r2']:.4f}
    • 方向准确率:              {metrics['direction_accuracy']:.4f}
    • 平均不确定性:            {metrics['mean_uncertainty']:.6f}
    
    数据统计:
    ----------------------------------------
    • 测试样本数量:            {len(predictions)}
    • 预测值范围:              [{predictions.min():.4f}, {predictions.max():.4f}]
    • 真实值范围:              [{targets.min():.4f}, {targets.max():.4f}]
    • 不确定性范围:            [{uncertainties.min():.4f}, {uncertainties.max():.4f}]
    
    模型特点:
    ----------------------------------------
    • ✅ 贝叶斯神经网络提供不确定性估计
    • ✅ 图神经网络捕捉股票间关系
    • ✅ 技术指标和价格特征融合
    • ✅ 变分推理实现参数不确定性
    • ✅ KL散度正则化防止过拟合
    
    使用建议:
    ----------------------------------------
    1. 高不确定性预测需要谨慎对待
    2. 结合传统分析方法使用
    3. 定期重训练模型以适应市场变化
    4. 监控模型性能指标的变化
    
    ========================================
    """
    
    print(report)
    
    # 保存报告到文件
    with open('dbgnn_prediction_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("📄 报告已保存为 'dbgnn_prediction_report.txt'")


def main():
    """主演示函数"""
    
    print("🎯 开始DBGNN完整演示流程\n")
    
    try:
        # 1. 快速演示：数据生成和可视化
        generator, price_data, features_data = quick_demo()
        
        # 2. 模型训练
        trainer, test_loader, history = train_demo_model(generator, price_data, features_data)
        
        # 3. 结果分析
        metrics, predictions, targets, uncertainties = analyze_predictions(trainer, test_loader)
        
        # 4. 交互式可视化
        create_interactive_visualization(generator, price_data, predictions, targets, uncertainties)
        
        # 5. 生成报告
        generate_prediction_report(metrics, predictions, targets, uncertainties)
        
        print("\n🎉 演示完成！")
        print("\n📁 生成的文件:")
        print("   • demo_results_analysis.png - 结果分析图")
        print("   • dbgnn_interactive_dashboard.html - 交互式Dashboard")
        print("   • dbgnn_prediction_report.txt - 预测报告")
        print("   • demo_logs/ - 训练日志")
        print("   • demo_models/ - 训练好的模型")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()