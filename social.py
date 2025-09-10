import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pysr import PySRRegressor
import warnings
import os
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# 动态获取数据集信息
data_file_path = '/home/xyh/PySR/data/D5.txt'
dataset_name = os.path.basename(data_file_path).split('.')[0]  # 从文件名提取数据集名称

# 读取数据
data = pd.read_csv(data_file_path, sep='\t')

print(f"数据集: {dataset_name}")
print(f"原始数据形状: {data.shape}")
print(f"数据列名: {list(data.columns)}")
print("\n数据预览:")
print(data.head(10))

# 动态获取特征列（排除目标变量y）
feature_columns = [col for col in data.columns if col != 'y']
if 'country' in feature_columns:
    feature_columns.remove('country')
if 'year' in feature_columns:
    feature_columns.remove('year')

# 准备特征和目标变量
X = data[feature_columns].values
y = data['y'].values

print(f"\n特征矩阵形状: {X.shape}")
print(f"目标变量形状: {y.shape}")
print(f"总样本数: {len(data)}")
print(f"特征维度: {', '.join(feature_columns)}")
print(f"目标变量: y")

print(f"y - 均值: {np.mean(y):.4f}, 标准差: {np.std(y):.4f}")

# 划分训练集和测试集 (75% 训练，25% 测试)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=None
)

# 对输入特征进行标准化
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# 对目标变量进行标准化
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

print(f"\n数据划分:")
print(f"训练集大小: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"测试集大小: {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"输入特征已标准化")
print(f"目标变量已标准化")

# 定义计算指标的函数
def calculate_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    return mse, r2

# 根据特征数量动态调整复杂度范围
num_features = len(feature_columns)
if num_features <= 3:
    complexity_min, complexity_max = 5, 15
elif num_features <= 5:
    complexity_min, complexity_max = 30, 50
else:
    complexity_min, complexity_max = 8, 20

print(f"\n根据特征数量({num_features})设置复杂度范围: {complexity_min}-{complexity_max}")

# 模型参数配置
model_params = {
    "model_selection": "best",
    "niterations": 50,
    "binary_operators": ["+", "*"],
    "unary_operators": [
        "exp", "log", "sqrt",
        "square(x) = x^2",
        "inv(x) = 1/x",
    ],
    "extra_sympy_mappings": {
        "square": lambda x: x**2,
        "inv": lambda x: 1 / x,
    },
    "populations": 30,
    "population_size": 50,
    "maxsize": 30,
    "should_optimize_constants": True,
    "optimizer_algorithm": "BFGS",
    "optimizer_nrestarts": 3,
    "weight_simplify": 0.01,
    "weight_optimize": 0.1,
    "parsimony": 0.01,
    "tournament_selection_n": 10,
    "tournament_selection_p": 0.9,
    "precision": 32,
    "verbosity": 1
}

# 创建结果目录
results_dir = "/home/xyh/PySR/result"
os.makedirs(results_dir, exist_ok=True)

# 存储所有运行结果
all_results = []

print("\n=== 开始10次重复实验 ===")
for seed in range(1, 11):
    print(f"\n--- 第 {seed} 次运行 (随机种子: {seed}) ---")
    
    # 设置随机种子
    model_params["random_state"] = seed
    
    # 创建模型
    model = PySRRegressor(**model_params)
    
    print(f"开始符号回归搜索 (种子: {seed})...")
    
    # 训练模型
    model.fit(X_train_scaled, y_train_scaled)
    
    # 预测
    y_test_pred_scaled = model.predict(X_test_scaled)
    
    # 反标准化预测结果
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
    
    # 计算测试集指标
    test_mse, test_r2 = calculate_metrics(y_test, y_test_pred)
    
    print(f"测试集 MSE: {test_mse:.6f}, R²: {test_r2:.6f}")
    
    # 筛选指定复杂度范围的方程
    filtered_equations = []
    if hasattr(model, 'equations_') and model.equations_ is not None:
        for idx, row in model.equations_.iterrows():
            complexity = int(row['complexity'])
            if complexity_min <= complexity <= complexity_max:
                # 获取方程的符号表达式
                equation_sympy = model.get_best(index=idx)
                
                # 计算该方程在测试集上的性能
                y_test_pred_scaled_eq = model.predict(X_test_scaled, index=idx)
                y_test_pred_eq = scaler_y.inverse_transform(y_test_pred_scaled_eq.reshape(-1, 1)).flatten()
                
                # 计算该方程的测试集指标
                eq_test_mse, eq_test_r2 = calculate_metrics(y_test, y_test_pred_eq)
                
                equation_result = {
                    "expression": str(row['equation']),
                    "complexity": complexity,
                    "test_mse": float(eq_test_mse),
                    "test_r2": float(eq_test_r2)
                }
                filtered_equations.append(equation_result)
    
    # 保存本次运行结果
    if filtered_equations:
        run_result = {
            "run_id": seed,
            "equations": filtered_equations
        }
        all_results.append(run_result)
        print(f"找到 {len(filtered_equations)} 个复杂度{complexity_min}-{complexity_max}的方程")
    else:
        print(f"未找到复杂度{complexity_min}-{complexity_max}的方程")

print("\n=== 10次运行完成 ===")

# 统计总结果
total_equations = sum(len(result["equations"]) for result in all_results)
print(f"总共找到 {total_equations} 个复杂度{complexity_min}-{complexity_max}的方程")

# 生成时间戳
timestamp = datetime.now().strftime("%m%d%H%M")

# 动态生成实验名称和文件名
experiment_name = f"{dataset_name}"
result_file = os.path.join(results_dir, f"{experiment_name}_{timestamp}.txt")

# 保存结果到文件
with open(result_file, 'w', encoding='utf-8') as f:
    f.write(f"数据集: {dataset_name}\n")
    f.write(f"特征变量: {', '.join(feature_columns)}\n")
    f.write(f"目标变量: y\n")
    f.write(f"实验名称: {experiment_name}\n")
    f.write(f"复杂度范围: {complexity_min}-{complexity_max}\n")
    f.write(f"总运行次数: 10\n")
    f.write(f"找到的方程总数: {total_equations}\n")
    f.write(f"时间戳: {timestamp}\n")
    f.write("\n" + "="*50 + "\n")
    
    for result in all_results:
        f.write(f"\n第 {result['run_id']} 次运行:\n")
        f.write(f"找到 {len(result['equations'])} 个方程\n")
        f.write("-" * 30 + "\n")
        
        for i, eq in enumerate(result['equations'], 1):
            # 格式化常数为一位小数
            import re
            expression = eq['expression']
            # 将浮点数格式化为一位小数
            expression = re.sub(r'\b\d+\.\d+\b', lambda m: f"{float(m.group()):.1f}", expression)
            
            f.write(f"方程 {i}:\n")
            f.write(f"  表达式: {expression}\n")
            f.write(f"  复杂度: {eq['complexity']}\n")
            f.write(f"  测试集MSE: {eq['test_mse']:.6f}\n")
            f.write(f"  测试集R²: {eq['test_r2']:.6f}\n")
            f.write("\n")

print(f"\n=== 结果保存完成 ===")
print(f"数据集: {dataset_name}")
print(f"特征变量: {', '.join(feature_columns)}")
print(f"复杂度范围: {complexity_min}-{complexity_max}")
print(f"结果已保存到: {result_file}")
print(f"包含 {len(all_results)} 次有效运行的复杂度{complexity_min}-{complexity_max}方程")
print(f"总方程数: {total_equations}")