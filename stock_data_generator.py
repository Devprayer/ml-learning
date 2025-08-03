import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import networkx as nx
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random


class StockDataGenerator:
    """股票数据生成器，用于生成模拟股票价格和关系图"""
    
    def __init__(self, 
                 num_stocks: int = 50,
                 time_periods: int = 252,  # 一年的交易日
                 seed: int = 42):
        self.num_stocks = num_stocks
        self.time_periods = time_periods
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 生成股票代码
        self.stock_symbols = [f"STOCK_{i:03d}" for i in range(num_stocks)]
        
        # 行业分类
        self.sectors = ['Technology', 'Finance', 'Healthcare', 'Energy', 'Consumer', 'Industrial']
        self.stock_sectors = {stock: random.choice(self.sectors) for stock in self.stock_symbols}
        
    def generate_correlation_matrix(self, 
                                  sector_correlation: float = 0.3,
                                  random_correlation: float = 0.1) -> np.ndarray:
        """生成股票间的相关性矩阵"""
        correlation_matrix = np.eye(self.num_stocks)
        
        for i in range(self.num_stocks):
            for j in range(i + 1, self.num_stocks):
                # 同行业股票有更高的相关性
                if self.stock_sectors[self.stock_symbols[i]] == self.stock_sectors[self.stock_symbols[j]]:
                    corr = np.random.normal(sector_correlation, 0.1)
                else:
                    corr = np.random.normal(0, random_correlation)
                
                # 确保相关性在合理范围内
                corr = np.clip(corr, -0.8, 0.8)
                correlation_matrix[i, j] = corr
                correlation_matrix[j, i] = corr
        
        # 确保正定性
        eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
        eigenvals = np.maximum(eigenvals, 0.01)
        correlation_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        # 重新标准化对角线
        d_inv = 1.0 / np.sqrt(np.diag(correlation_matrix))
        correlation_matrix = np.outer(d_inv, d_inv) * correlation_matrix
        
        return correlation_matrix
    
    def generate_stock_prices(self, 
                            initial_price: float = 100.0,
                            volatility: float = 0.02,
                            drift: float = 0.0005) -> pd.DataFrame:
        """使用几何布朗运动生成股票价格"""
        
        # 生成相关性矩阵
        correlation_matrix = self.generate_correlation_matrix()
        
        # 生成随机冲击
        random_shocks = np.random.multivariate_normal(
            mean=np.zeros(self.num_stocks),
            cov=correlation_matrix,
            size=self.time_periods
        )
        
        # 初始化价格矩阵
        prices = np.zeros((self.time_periods + 1, self.num_stocks))
        prices[0] = initial_price
        
        # 为每个股票设定不同的参数
        stock_volatilities = np.random.normal(volatility, volatility * 0.3, self.num_stocks)
        stock_drifts = np.random.normal(drift, drift * 0.5, self.num_stocks)
        
        # 生成价格路径
        for t in range(1, self.time_periods + 1):
            for i in range(self.num_stocks):
                # 几何布朗运动
                dt = 1  # 一天
                price_change = (stock_drifts[i] * dt + 
                              stock_volatilities[i] * np.sqrt(dt) * random_shocks[t-1, i])
                prices[t, i] = prices[t-1, i] * np.exp(price_change)
        
        # 创建日期索引
        start_date = datetime(2023, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(self.time_periods + 1)]
        
        # 创建DataFrame
        price_df = pd.DataFrame(
            prices,
            index=dates,
            columns=self.stock_symbols
        )
        
        return price_df
    
    def calculate_technical_indicators(self, price_df: pd.DataFrame, 
                                     window_short: int = 5,
                                     window_long: int = 20) -> pd.DataFrame:
        """计算技术指标"""
        features = pd.DataFrame(index=price_df.index)
        
        for stock in self.stock_symbols:
            prices = price_df[stock]
            
            # 收益率
            features[f'{stock}_return'] = prices.pct_change()
            
            # 移动平均
            features[f'{stock}_ma_short'] = prices.rolling(window_short).mean()
            features[f'{stock}_ma_long'] = prices.rolling(window_long).mean()
            
            # 相对强弱指数 (RSI)
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features[f'{stock}_rsi'] = 100 - (100 / (1 + rs))
            
            # 波动率
            features[f'{stock}_volatility'] = prices.pct_change().rolling(window_short).std()
            
            # 价格相对位置
            high_window = prices.rolling(window_long).max()
            low_window = prices.rolling(window_long).min()
            features[f'{stock}_price_position'] = (prices - low_window) / (high_window - low_window)
        
        return features.fillna(0)
    
    def create_stock_graph(self, correlation_threshold: float = 0.2) -> Data:
        """创建股票关系图"""
        correlation_matrix = self.generate_correlation_matrix()
        
        # 创建邻接矩阵
        adj_matrix = np.abs(correlation_matrix) > correlation_threshold
        np.fill_diagonal(adj_matrix, False)  # 移除自连接
        
        # 创建边列表
        edge_list = []
        edge_weights = []
        
        for i in range(self.num_stocks):
            for j in range(i + 1, self.num_stocks):
                if adj_matrix[i, j]:
                    edge_list.append([i, j])
                    edge_list.append([j, i])  # 无向图
                    weight = correlation_matrix[i, j]
                    edge_weights.extend([weight, weight])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
        
        # 节点特征（行业编码）
        sector_to_idx = {sector: idx for idx, sector in enumerate(self.sectors)}
        node_features = []
        
        for stock in self.stock_symbols:
            sector_vec = [0] * len(self.sectors)
            sector_vec[sector_to_idx[self.stock_sectors[stock]]] = 1
            node_features.append(sector_vec)
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def prepare_time_series_data(self, 
                               price_df: pd.DataFrame, 
                               features_df: pd.DataFrame,
                               sequence_length: int = 10,
                               prediction_horizon: int = 1) -> List[Data]:
        """准备时间序列数据用于训练"""
        
        data_list = []
        base_graph = self.create_stock_graph()
        
        # 标准化特征
        feature_columns = [col for col in features_df.columns if '_return' in col or 
                          '_ma_' in col or '_rsi' in col or '_volatility' in col or '_price_position' in col]
        
        for t in range(sequence_length, len(price_df) - prediction_horizon):
            # 获取历史特征
            historical_features = []
            for stock_idx, stock in enumerate(self.stock_symbols):
                stock_features = []
                
                # 价格特征
                price_window = price_df[stock].iloc[t-sequence_length:t].values
                price_features = [
                    np.mean(price_window),
                    np.std(price_window),
                    price_window[-1] / price_window[0] - 1,  # 总收益率
                ]
                
                # 技术指标特征
                tech_features = []
                for feature_col in feature_columns:
                    if feature_col.startswith(f'{stock}_'):
                        tech_features.append(features_df[feature_col].iloc[t-1])
                
                stock_features.extend(price_features)
                stock_features.extend(tech_features)
                historical_features.append(stock_features)
            
            # 节点特征包含行业信息和历史特征
            sector_features = base_graph.x.numpy()
            historical_features = np.array(historical_features)
            
            # 组合特征
            node_features = np.concatenate([sector_features, historical_features], axis=1)
            x = torch.tensor(node_features, dtype=torch.float)
            
            # 目标值（未来收益率）
            targets = []
            for stock in self.stock_symbols:
                current_price = price_df[stock].iloc[t]
                future_price = price_df[stock].iloc[t + prediction_horizon]
                return_rate = (future_price - current_price) / current_price
                targets.append(return_rate)
            
            y = torch.tensor(targets, dtype=torch.float).unsqueeze(1)
            
            # 创建图数据
            graph_data = Data(
                x=x,
                edge_index=base_graph.edge_index,
                edge_attr=base_graph.edge_attr,
                y=y
            )
            
            data_list.append(graph_data)
        
        return data_list
    
    def visualize_stock_network(self, correlation_threshold: float = 0.3):
        """可视化股票关系网络"""
        correlation_matrix = self.generate_correlation_matrix()
        
        # 创建网络图
        G = nx.Graph()
        
        # 添加节点
        for i, stock in enumerate(self.stock_symbols):
            G.add_node(i, name=stock, sector=self.stock_sectors[stock])
        
        # 添加边
        for i in range(self.num_stocks):
            for j in range(i + 1, self.num_stocks):
                if abs(correlation_matrix[i, j]) > correlation_threshold:
                    G.add_edge(i, j, weight=correlation_matrix[i, j])
        
        # 绘制网络
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # 按行业着色
        sector_colors = plt.cm.Set3(np.linspace(0, 1, len(self.sectors)))
        sector_color_map = {sector: color for sector, color in zip(self.sectors, sector_colors)}
        
        node_colors = [sector_color_map[self.stock_sectors[self.stock_symbols[node]]] 
                      for node in G.nodes()]
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, alpha=0.8)
        
        # 绘制边
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        edge_colors = ['red' if w < 0 else 'blue' for w in weights]
        edge_widths = [abs(w) * 3 for w in weights]
        
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, alpha=0.6)
        
        # 添加标签
        labels = {i: stock.split('_')[1] for i, stock in enumerate(self.stock_symbols)}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        plt.title(f"Stock Correlation Network (threshold={correlation_threshold})")
        
        # 添加图例
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=sector_color_map[sector], 
                                    markersize=10, label=sector) 
                         for sector in self.sectors]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def plot_sample_stocks(self, price_df: pd.DataFrame, num_samples: int = 5):
        """绘制样本股票价格走势"""
        sample_stocks = np.random.choice(self.stock_symbols, num_samples, replace=False)
        
        plt.figure(figsize=(15, 8))
        for stock in sample_stocks:
            plt.plot(price_df.index, price_df[stock], label=stock, linewidth=2)
        
        plt.title("Sample Stock Price Movements")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def get_data_statistics(self, price_df: pd.DataFrame, features_df: pd.DataFrame):
        """获取数据统计信息"""
        print("=== 股票数据统计 ===")
        print(f"股票数量: {self.num_stocks}")
        print(f"时间期数: {self.time_periods}")
        print(f"行业分布: {dict(pd.Series(list(self.stock_sectors.values())).value_counts())}")
        
        print("\n=== 价格统计 ===")
        returns = price_df.pct_change().dropna()
        print(f"平均收益率: {returns.mean().mean():.4f}")
        print(f"收益率标准差: {returns.std().mean():.4f}")
        print(f"最大收益率: {returns.max().max():.4f}")
        print(f"最小收益率: {returns.min().min():.4f}")
        
        print("\n=== 特征统计 ===")
        print(f"特征维度: {features_df.shape[1]}")
        print(f"特征类型数量: {len([col for col in features_df.columns if '_return' in col])//self.num_stocks}")


if __name__ == "__main__":
    # 示例使用
    generator = StockDataGenerator(num_stocks=30, time_periods=252)
    
    # 生成股票价格
    price_data = generator.generate_stock_prices()
    
    # 计算技术指标
    features_data = generator.calculate_technical_indicators(price_data)
    
    # 准备图数据
    graph_data_list = generator.prepare_time_series_data(price_data, features_data)
    
    # 显示统计信息
    generator.get_data_statistics(price_data, features_data)
    
    # 可视化
    generator.plot_sample_stocks(price_data)
    generator.visualize_stock_network()
    
    print(f"\n生成了 {len(graph_data_list)} 个图数据样本")
    print(f"每个图的节点特征维度: {graph_data_list[0].x.shape}")
    print(f"每个图的边数量: {graph_data_list[0].edge_index.shape[1]}")
    print(f"目标维度: {graph_data_list[0].y.shape}")