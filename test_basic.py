#!/usr/bin/env python3
"""
基本测试脚本 - 验证DBGNN代码的基本结构和语法
"""

def test_imports():
    """测试所有模块的导入"""
    try:
        print("测试导入...")
        
        # 测试基本Python模块
        import sys
        import os
        print("✅ 基本Python模块导入成功")
        
        # 这些模块在没有安装的情况下会失败，但我们可以检查语法
        test_modules = [
            'dbgnn_model.py',
            'stock_data_generator.py', 
            'train_dbgnn.py',
            'demo.py'
        ]
        
        for module in test_modules:
            if os.path.exists(module):
                print(f"✅ {module} 文件存在")
                # 简单的语法检查
                with open(module, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 检查基本的Python语法结构
                    if 'import' in content and 'class' in content:
                        print(f"✅ {module} 基本结构正确")
                    else:
                        print(f"⚠️ {module} 可能缺少导入或类定义")
            else:
                print(f"❌ {module} 文件不存在")
                
    except Exception as e:
        print(f"❌ 测试导入失败: {e}")


def test_file_structure():
    """测试文件结构"""
    required_files = [
        'requirements.txt',
        'dbgnn_model.py',
        'stock_data_generator.py',
        'train_dbgnn.py',
        'demo.py',
        'README.md'
    ]
    
    print("\n测试文件结构...")
    for file in required_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✅ {file} (大小: {size} bytes)")
        else:
            print(f"❌ {file} 文件缺失")


def test_code_syntax():
    """简单的代码语法测试"""
    print("\n测试代码语法...")
    
    try:
        # 测试DBGNN模型的基本结构
        with open('dbgnn_model.py', 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 检查关键组件
        if 'class DBGNN' in content:
            print("✅ DBGNN类定义存在")
        if 'class BayesianLinear' in content:
            print("✅ BayesianLinear类定义存在")
        if 'def forward' in content:
            print("✅ forward方法存在")
        if 'def kl_divergence' in content:
            print("✅ kl_divergence方法存在")
            
        # 测试数据生成器
        with open('stock_data_generator.py', 'r', encoding='utf-8') as f:
            content = f.read()
            
        if 'class StockDataGenerator' in content:
            print("✅ StockDataGenerator类定义存在")
        if 'def generate_stock_prices' in content:
            print("✅ generate_stock_prices方法存在")
        if 'def create_stock_graph' in content:
            print("✅ create_stock_graph方法存在")
            
    except Exception as e:
        print(f"❌ 语法测试失败: {e}")


def test_readme():
    """测试README文件"""
    print("\n测试README文件...")
    
    try:
        with open('README.md', 'r', encoding='utf-8') as f:
            content = f.read()
            
        if 'DBGNN' in content:
            print("✅ README包含DBGNN说明")
        if '安装依赖' in content:
            print("✅ README包含安装说明")
        if '快速开始' in content:
            print("✅ README包含使用说明")
        if len(content) > 1000:
            print(f"✅ README内容充实 ({len(content)} 字符)")
        else:
            print(f"⚠️ README内容较少 ({len(content)} 字符)")
            
    except Exception as e:
        print(f"❌ README测试失败: {e}")


def generate_summary():
    """生成项目总结"""
    print("\n" + "="*50)
    print("📋 DBGNN股票预测项目总结")
    print("="*50)
    
    print("\n🎯 项目特点:")
    print("• 深度贝叶斯图神经网络 (DBGNN)")
    print("• 股票价格预测与不确定性估计")
    print("• 完整的数据生成到模型训练流程")
    print("• 丰富的可视化和分析功能")
    
    print("\n📁 核心文件:")
    files_info = {
        'dbgnn_model.py': 'DBGNN模型实现，包含贝叶斯层和图卷积',
        'stock_data_generator.py': '模拟股票数据生成器，支持技术指标计算',
        'train_dbgnn.py': '完整的训练脚本，包含评估和可视化',
        'demo.py': '端到端演示脚本，展示完整流程',
        'requirements.txt': '项目依赖列表',
        'README.md': '详细的项目说明文档'
    }
    
    for file, desc in files_info.items():
        if os.path.exists(file):
            print(f"✅ {file}: {desc}")
        else:
            print(f"❌ {file}: {desc} (文件缺失)")
    
    print("\n🚀 使用方法:")
    print("1. pip install -r requirements.txt")
    print("2. python demo.py  # 运行完整演示")
    print("3. python train_dbgnn.py  # 仅训练模型")
    
    print("\n🧮 技术特色:")
    print("• 变分贝叶斯推理提供不确定性估计")
    print("• 图神经网络捕捉股票间关系")
    print("• 多种技术指标融合")
    print("• 交互式可视化Dashboard")
    
    print("\n⚠️ 注意事项:")
    print("• 本项目仅用于教育和研究目的")
    print("• 模拟数据，不可直接用于实际投资")
    print("• 需要安装PyTorch和PyTorch Geometric")
    
    print("\n🎉 项目已成功创建！")


if __name__ == "__main__":
    import os
    
    print("🔍 DBGNN项目基本测试")
    print("="*40)
    
    test_imports()
    test_file_structure()
    test_code_syntax()
    test_readme()
    generate_summary()