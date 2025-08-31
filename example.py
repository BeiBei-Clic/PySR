import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.linalg import qr
import warnings
warnings.filterwarnings('ignore')

# 生成数据
X = 2 * np.random.randn(100, 5)
y = 2.5382 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 0.5

from pysr import PySRRegressor
from pysr import jl

# 将数据传递给Julia环境，供自定义损失函数使用
jl.seval("global X_data, y_data")
jl.X_data = X
jl.y_data = y

# 在Julia中定义多重线性性检测和优化函数
jl.seval("""
using LinearAlgebra
using Statistics

# 检测多重线性性并优化系数的函数
function optimize_linear_terms(prediction, target, tree, dataset, options)
    # 基础损失
    base_loss = mean((prediction .- target) .^ 2)
    
    # 如果预测失败，返回无穷大损失
    if any(isnan.(prediction)) || any(isinf.(prediction))
        return Inf
    end
    
    # 尝试检测加法结构并优化
    try
        # 检查表达式是否为加法形式
        if tree.degree == 2 && tree.op == 1  # 假设操作符1是加法
            # 递归提取所有加法项
            terms = extract_additive_terms(tree, dataset, options)
            
            if length(terms) > 1
                # 构建项矩阵
                terms_matrix = hcat(terms...)
                
                # 检测多重线性性
                redundant_indices, remaining_indices = detect_multicollinearity(terms_matrix, target)
                
                if !isempty(redundant_indices)
                    # 优化剩余项的系数
                    if !isempty(remaining_indices)
                        remaining_terms = terms_matrix[:, remaining_indices]
                        optimized_coeffs = remaining_terms \\ target  # 最小二乘解
                        
                        # 计算优化后的预测
                        optimized_pred = remaining_terms * optimized_coeffs
                        optimized_loss = mean((optimized_pred .- target) .^ 2)
                        
                        # 修改复杂度奖励：使用乘法因子而不是减法，确保非负
                        complexity_reward_factor = 1.0 - (length(redundant_indices) * 0.05)
                        complexity_reward_factor = max(complexity_reward_factor, 0.1)  # 确保至少保留10%
                        
                        return optimized_loss * complexity_reward_factor
                    end
                end
            end
        end
    catch
        # 如果优化失败，返回基础损失
    end
    
    return base_loss
end

# 提取加法项的递归函数
# 提取加法项的递归函数（支持复杂组合项）
function extract_additive_terms(tree, dataset, options)
    terms = Vector{Vector{Float64}}()
    
    # 递归提取加法项
    extract_terms_recursive!(terms, tree, dataset, options)
    
    return terms
end

# 递归提取函数
function extract_terms_recursive!(terms, tree, dataset, options)
    if tree.degree == 2 && tree.op == 1  # 加法操作
        # 递归处理左右子树
        extract_terms_recursive!(terms, tree.l, dataset, options)
        extract_terms_recursive!(terms, tree.r, dataset, options)
    else
        # 非加法节点：作为完整的一项处理
        term_value, success = eval_tree_array(tree, dataset.X, options)
        if success && !any(isnan.(term_value)) && !any(isinf.(term_value))
            push!(terms, term_value)
        end
    end
end

# 检测多重线性性
function detect_multicollinearity(terms_matrix, target; correlation_threshold=0.9)
    n_terms = size(terms_matrix, 2)
    
    if n_terms <= 1
        return Int[], collect(1:n_terms)
    end
    
    # 标准化
    terms_scaled = (terms_matrix .- mean(terms_matrix, dims=1)) ./ std(terms_matrix, dims=1)
    
    # 计算相关系数矩阵
    corr_matrix = cor(terms_scaled)
    
    redundant_indices = Set{Int}()
    
    # 检测高度相关的项对
    for i in 1:n_terms
        for j in (i+1):n_terms
            if abs(corr_matrix[i, j]) > correlation_threshold
                # 计算与目标的相关性，保留更重要的项
                importance_i = abs(cor(terms_matrix[:, i], target))
                importance_j = abs(cor(terms_matrix[:, j], target))
                
                if importance_i < importance_j
                    push!(redundant_indices, i)
                else
                    push!(redundant_indices, j)
                end
            end
        end
    end
    
    # 使用QR分解检测线性相关性
    try
        Q, R = qr(terms_scaled)
        diagonal = abs.(diag(R))
        rank_deficient = findall(x -> x < 1e-10, diagonal)
        union!(redundant_indices, rank_deficient)
    catch
    end
    
    redundant_list = collect(redundant_indices)
    remaining_list = setdiff(1:n_terms, redundant_indices)
    
    return redundant_list, remaining_list
end
""")

# 创建增强的PySR模型
model = PySRRegressor(
    model_selection="best",
    niterations=40,
    binary_operators=["+", "*"],
    unary_operators=[
        "cos",
        "exp",
        "sin",
        "inv(x) = 1/x",
    ],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    # 使用自定义损失函数，在每次个体评估时进行多重线性性优化
    loss_function="""
    function enhanced_loss(tree, dataset, options)
        # 评估树得到预测值
        prediction, flag = eval_tree_array(tree, dataset.X, options)
        
        if !flag
            return Inf
        end
        
        # 调用多重线性性优化函数
        return optimize_linear_terms(prediction, dataset.y, tree, dataset, options)
    end
    """,
    # 优化相关参数
    populations=20,
    population_size=35,
    maxsize=20,  # 限制复杂度
    should_optimize_constants=True,
    optimizer_algorithm="BFGS",
    optimizer_nrestarts=2,
    # 增加变异和简化的权重，促进结构优化
    weight_simplify=0.01,
    weight_optimize=0.1,
    # 解析性偏好，鼓励简单结构
    parsimony=0.05,
    # 提高选择压力
    tournament_selection_n=12,
    tournament_selection_p=0.95,
    verbosity=1
)

print("开始增强的符号回归搜索（实时多重线性性优化）...")
model.fit(X, y)

print("\n=== 搜索结果 ===")
print(model)

# 性能评估
print("\n=== 性能评估 ===")
predictions = model.predict(X)
mse = np.mean((predictions - y) ** 2)
r2 = 1 - (np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2))
print(f"均方误差 (MSE): {mse:.6f}")
print(f"决定系数 (R²): {r2:.6f}")

# 显示学到的最佳方程
print(f"\n学到的最佳方程: {model.sympy()}")
print(f"真实关系: y = 2.5382 * cos(x3) + x0² - 0.5")

# 分析方程复杂度
if hasattr(model, 'equations_') and model.equations_ is not None:
    best_eq = model.equations_.iloc[-1]  # 最佳方程通常在最后
    print(f"\n最佳方程复杂度: {best_eq['complexity']}")
    print(f"最佳方程损失: {best_eq['loss']:.6f}")
    
    # 显示所有候选方程的复杂度分布
    print("\n=== 复杂度分析 ===")
    complexities = model.equations_['complexity'].values
    losses = model.equations_['loss'].values
    
    print(f"平均复杂度: {np.mean(complexities):.2f}")
    print(f"最小复杂度: {np.min(complexities)}")
    print(f"最大复杂度: {np.max(complexities)}")
    print(f"复杂度标准差: {np.std(complexities):.2f}")
    
    # 显示帕累托前沿（复杂度vs损失）
    print("\n=== 帕累托前沿（复杂度 vs 损失）===")
    for i, (comp, loss) in enumerate(zip(complexities, losses)):
        equation = model.equations_.iloc[i]['equation']
        print(f"复杂度 {comp:2d}: 损失 {loss:.6f} - {equation}")
