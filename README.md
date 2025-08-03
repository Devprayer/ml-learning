# DBGNN 股票预测模型

这是一个基于深度贝叶斯图神经网络（Deep Bayesian Graph Neural Network, DBGNN）的股票价格预测模型。该模型结合了图神经网络和贝叶斯深度学习，不仅能进行股票收益率预测，还能提供预测的不确定性估计。

## 🎯 项目特点

- **🧠 深度贝叶斯网络**: 使用变分推理提供预测不确定性
- **🕸️ 图神经网络**: 捕捉股票间的相关性和依赖关系
- **📊 技术指标融合**: 整合多种技术分析指标
- **🎨 丰富可视化**: 提供静态和交互式图表
- **📱 完整演示**: 从数据生成到模型训练的完整流程

## 🏗️ 项目结构

```
├── requirements.txt           # 项目依赖
├── dbgnn_model.py            # DBGNN模型实现
├── stock_data_generator.py   # 股票数据生成器
├── train_dbgnn.py           # 模型训练脚本
├── demo.py                  # 完整演示脚本
└── README.md               # 项目说明文档
```

## 📦 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖包括：
- PyTorch (>=2.0.0)
- PyTorch Geometric (>=2.3.0)
- NumPy, Pandas, Matplotlib
- Seaborn, Plotly, NetworkX
- Scikit-learn, SciPy

## 🚀 快速开始

### 1. 运行完整演示

```bash
python demo.py
```

这将执行完整的演示流程，包括：
- 生成模拟股票数据
- 训练DBGNN模型
- 进行预测和不确定性分析
- 生成可视化报告

### 2. 仅训练模型

```bash
python train_dbgnn.py
```

### 3. 仅生成数据

```python
from stock_data_generator import StockDataGenerator

# 创建数据生成器
generator = StockDataGenerator(num_stocks=30, time_periods=252)

# 生成股票价格
price_data = generator.generate_stock_prices()

# 计算技术指标
features_data = generator.calculate_technical_indicators(price_data)

# 准备训练数据
graph_data_list = generator.prepare_time_series_data(price_data, features_data)
```

## 🧮 模型架构

### DBGNN 模型组件

1. **图卷积层**: 使用GCN捕捉股票间关系
2. **贝叶斯全连接层**: 提供参数不确定性估计
3. **变分推理**: 使用重参数化技巧进行训练
4. **不确定性量化**: 区分认知不确定性和偶然不确定性

### 损失函数

```
Total Loss = MSE Loss + β × KL Divergence
```

其中：
- MSE Loss: 预测准确性
- KL Divergence: 贝叶斯正则化项
- β: KL权重超参数

## 📊 数据特征

### 股票特征
- 价格统计特征（均值、标准差、收益率）
- 技术指标（MA、RSI、波动率等）
- 行业分类信息
- 价格相对位置

### 图结构
- 节点：股票
- 边：基于相关性的连接
- 边权重：相关性强度
- 行业聚类：同行业股票更高连接度

## 📈 评估指标

- **MSE (均方误差)**: 预测准确性
- **MAE (平均绝对误差)**: 鲁棒性指标
- **R² (决定系数)**: 解释能力
- **方向准确率**: 涨跌方向预测准确性
- **不确定性校准**: 预测置信度质量

## 🎨 可视化功能

### 1. 静态图表
- 股票价格走势图
- 相关性网络图
- 预测vs真实值散点图
- 不确定性分布直方图
- 训练历史曲线

### 2. 交互式Dashboard
- Plotly交互式图表
- 性能指标表格
- 鼠标悬停信息
- 缩放和选择功能

## ⚙️ 超参数配置

### 模型参数
```python
DBGNN(
    input_dim=特征维度,
    hidden_dim=64,           # 隐藏层维度
    output_dim=1,            # 输出维度
    num_layers=3,            # GCN层数
    dropout=0.1,             # Dropout率
    prior_std=1.0            # 先验标准差
)
```

### 训练参数
```python
trainer.train(
    epochs=200,              # 训练轮数
    learning_rate=0.001,     # 学习率
    weight_decay=1e-5,       # 权重衰减
    kl_weight=1e-4,          # KL散度权重
    patience=20              # 早停耐心值
)
```

## 📋 输出文件

运行演示后将生成以下文件：

- `demo_results_analysis.png` - 详细结果分析图
- `dbgnn_interactive_dashboard.html` - 交互式仪表板
- `dbgnn_prediction_report.txt` - 预测性能报告
- `demo_logs/` - TensorBoard训练日志
- `demo_models/` - 训练好的模型文件

## 🔧 自定义使用

### 使用真实股票数据

```python
# 替换模拟数据生成器
import yfinance as yf

# 下载真实股票数据
tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
data = yf.download(tickers, start='2020-01-01', end='2023-12-31')

# 处理数据格式并训练模型
# ... 具体实现请参考stock_data_generator.py
```

### 调整网络结构

```python
# 自定义GCN层数和隐藏维度
model = DBGNN(
    input_dim=input_dim,
    hidden_dim=128,          # 增加容量
    num_layers=4,            # 增加深度
    dropout=0.2              # 增加正则化
)
```

## 📚 理论背景

### 贝叶斯神经网络
- 将网络权重视为概率分布
- 使用变分推理近似后验分布
- 提供预测不确定性量化

### 图神经网络
- 利用股票间相关性构建图结构
- 使用消息传递机制聚合邻居信息
- 捕捉复杂的市场关系模式

### 不确定性类型
- **认知不确定性**: 模型参数不确定性
- **偶然不确定性**: 数据噪声不确定性
- **总不确定性**: 两者结合

## ⚠️ 免责声明

本项目仅用于教育和研究目的。股票预测具有高度不确定性，本模型的预测结果不应作为投资建议。实际投资时请：

1. 结合多种分析方法
2. 考虑风险管理
3. 咨询专业投资顾问
4. 充分了解市场风险

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！

## 📄 许可证

MIT License - 详见LICENSE文件

---

**Happy Trading with AI! 🚀📈**
