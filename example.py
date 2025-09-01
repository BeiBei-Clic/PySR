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
