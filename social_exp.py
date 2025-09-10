import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pysr import PySRRegressor
import warnings
import os
import json
from datetime import datetime
import sympy as sp
import re
warnings.filterwarnings('ignore')

# 动态获取数据集信息
data_file_path = '/home/xyh/PySR/data/D2.txt'
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

# 对目标变量取指数变换
y = np.log(y)

print(f"\n特征矩阵形状: {X.shape}")
print(f"目标变量形状: {y.shape}")
print(f"总样本数: {len(data)}")
print(f"特征维度: {', '.join(feature_columns)}")
print(f"目标变量: exp(y)")

print(f"exp(y) - 均值: {np.mean(y):.4f}, 标准差: {np.std(y):.4f}")

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

# 定义数值格式化函数
def format_expression_numbers(expression_str):
    import re
    # 匹配浮点数的正则表达式
    def replace_number(match):
        num = float(match.group())
        if abs(num) < 1e-10:
            return '0'
        elif abs(num - round(num)) < 1e-10:
            return str(int(round(num)))
        else:
            return f'{num:.1f}'
    
    # 替换表达式中的所有数字
    formatted = re.sub(r'-?\d+\.\d+', replace_number, expression_str)
    return formatted

# 定义表达式化简函数
def simplify_expression(expression_str):
    symbols_dict = {
        'x0': sp.Symbol('x0'),
        'x1': sp.Symbol('x1'), 
        'x2': sp.Symbol('x2'),
        'x3': sp.Symbol('x3'),
        'x4': sp.Symbol('x4')
    }
    
    try:
        parsed = sp.sympify(expression_str, locals=symbols_dict)
    except:
        return f"解析错误: {expression_str}"
    
    # 使用温和的化简方法
    simplified = sp.expand(parsed)
    
    # 收集同类项时只使用符号对象
    symbol_list = [sp.Symbol(f'x{i}') for i in range(5)]
    simplified = sp.collect(simplified, symbol_list)
    
    # 数值验证
    test_values = {sp.Symbol(f'x{i}'): i + 1 for i in range(5)}
    try:
        original_val = float(parsed.subs(test_values))
        simplified_val = float(simplified.subs(test_values))
        if abs(original_val - simplified_val) > 1e-10:
            simplified = parsed
    except:
        simplified = parsed
    
    # 格式化数值
    def format_numbers(expr):
        if expr.is_number:
            val = float(expr)
            if abs(val) < 1e-10:
                return sp.Integer(0)
            elif abs(val - round(val)) < 1e-10:
                return sp.Integer(round(val))
            else:
                return sp.Float(val, 1)  # 改为1位小数
        elif expr.is_Add:
            return sp.Add(*[format_numbers(arg) for arg in expr.args])
        elif expr.is_Mul:
            return sp.Mul(*[format_numbers(arg) for arg in expr.args])
        elif expr.is_Pow:
            return sp.Pow(format_numbers(expr.base), format_numbers(expr.exp))
        else:
            return expr
    
    formatted = format_numbers(simplified)
    return str(formatted)

# 根据特征数量动态调整复杂度范围
num_features = len(feature_columns)
if num_features <= 3:
    complexity_min, complexity_max = 7, 30
elif num_features <= 5:
    complexity_min, complexity_max = 10, 50
else:
    complexity_min, complexity_max = 8, 20

print(f"\n根据特征数量({num_features})设置复杂度范围: {complexity_min}-{complexity_max}")

# 模型参数配置
# 在模型参数中添加自定义损失函数
model_params = {
    "model_selection": "best",
    "niterations": 100,
    "binary_operators": ["+", "*"],
    "unary_operators": [
        "exp", "log", "sqrt",
        "square(x) = x^2",
    ],
    "extra_sympy_mappings": {
        "square": lambda x: x**2,
    },
    "populations": 30,
    "population_size": 100,
    "maxsize": 30,
    "should_optimize_constants": True,
    "optimizer_algorithm": "BFGS",
    "optimizer_nrestarts": 3,
    "weight_simplify": 0.0005,
    "weight_optimize": 0.1,
    "parsimony": 0.0005,
    "tournament_selection_n": 10,
    "tournament_selection_p": 0.9,
    "precision": 32,
    "verbosity": 1,
    # 添加自定义损失函数约束常数为正值
    # "loss_function": """
    # function eval_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    #     # 检查负常数的函数
    #     is_negative_constant(node) = node.degree == 0 && node.constant && node.val::T < 0
    #     # 添加约束变量系数为正的函数
    #     is_negative_coefficient(node) = (
    #     # 检查乘法节点中的常数是否为负
    #     (node.op == 1 && node.l.degree == 0 && node.l.constant && node.l.val::T < 0) ||
    #     (node.op == 1 && node.r.degree == 0 && node.r.constant && node.r.val::T < 0) ||
    #     )
        
    #     # 计算负常数的数量
    #     num_negative_constants = count(is_negative_constant, tree)
        
    #     if num_negative_constants > 0
    #         # 对负常数施加惩罚
    #         return L(1000 * num_negative_constants)
    #     end
        
    #     # 正常的MSE损失计算
    #     prediction, flag = eval_tree_array(tree, dataset.X, options)
    #     if !flag
    #         return L(Inf)
    #     end
    #     return sum((prediction .- dataset.y) .^ 2) / dataset.n
    # end
    # """
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
    model = PySRRegressor(
        model_selection="best",
        niterations=50,
        binary_operators=["+", "*"],
        unary_operators=[
            "exp", "log", "sqrt",
            "square",
        ],
        extra_sympy_mappings={
            "square": lambda x: x**2,
        },
        populations=30,
        population_size=50,
        maxsize=30,
        should_optimize_constants=True,
        optimizer_algorithm="BFGS",
        optimizer_nrestarts=3,
        weight_simplify=0.0005,
        weight_optimize=0.1,
        parsimony=0.0005,
        tournament_selection_n=10,
        tournament_selection_p=0.9,
        precision=32,
        verbosity=1,
        random_state=seed,
        # 强制所有变量出现的约束
        constraints={
            # 确保每个变量至少出现一次
            "variable_names": [f"x{i}" for i in range(len(feature_columns))],
            "force_all_variables": True,
        },
        # 添加自定义损失函数约束正系数
        loss_function="""function eval_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
            # 检查是否所有变量都出现
            used_variables = Set{Int}()
            
            function collect_variables(node)
                if node.degree == 0 && !node.constant
                    push!(used_variables, node.feature)
                elseif node.degree == 1
                    collect_variables(node.l)
                elseif node.degree == 2
                    collect_variables(node.l)
                    collect_variables(node.r)
                end
            end
            
            collect_variables(tree)
            
            # 如果没有使用所有变量，施加巨大惩罚
            num_features = size(dataset.X, 1)
            if length(used_variables) < num_features
                return L(1e10)
            end
            
            # 检查负常数
            function has_negative_constant(node)
                if node.degree == 0 && node.constant && node.val < 0
                    return true
                elseif node.degree == 1
                    return has_negative_constant(node.l)
                elseif node.degree == 2
                    return has_negative_constant(node.l) || has_negative_constant(node.r)
                end
                return false
            end
            
            # 如果有负常数，施加惩罚
            if has_negative_constant(tree)
                return L(1e8)
            end
            
            # 正常的MSE损失计算
            prediction, flag = eval_tree_array(tree, dataset.X, options)
            if !flag
                return L(Inf)
            end
            return sum((prediction .- dataset.y) .^ 2) / dataset.n
        end"""
    )

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
                
                # 化简表达式
                original_expression = str(row['equation'])
                simplified_expression = simplify_expression(original_expression)
                
                equation_result = {
                    "original_expression": original_expression,
                    "simplified_expression": simplified_expression,
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
    f.write(f"总运行次数: 10\n")
    f.write(f"找到的方程总数: {total_equations}\n")
    f.write(f"时间戳: {timestamp}\n")
    f.write("\n" + "="*50 + "\n")
    
    for result in all_results:
        f.write(f"\n第 {result['run_id']} 次运行:\n")
        f.write(f"找到 {len(result['equations'])} 个方程\n")
        f.write("-" * 30 + "\n")
        
        for i, eq in enumerate(result['equations'], 1):
            f.write(f"方程 {i}:\n")
            f.write(f"  原始表达式: {eq['original_expression']}\n")
            f.write(f"  化简表达式: {eq['simplified_expression']}\n")
            f.write(f"  测试集MSE: {eq['test_mse']:.6f}\n")
            f.write(f"  测试集R²: {eq['test_r2']:.6f}\n")
            f.write("\n")

print(f"\n=== 结果保存完成 ===")
print(f"数据集: {dataset_name}")
print(f"特征变量: {', '.join(feature_columns)}")
print(f"结果已保存到: {result_file}")
print(f"包含 {len(all_results)} 次有效运行的方程")
print(f"总方程数: {total_equations}")
