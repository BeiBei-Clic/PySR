import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.linalg import qr
import warnings
warnings.filterwarnings('ignore')

from pysr import PySRRegressor
from pysr import jl

class EnhancedSymbolicRegression:
    """增强的符号回归算法，支持多重线性性检测和优化"""
    
    def __init__(self, test_size=0.2, random_state=42, **pysr_kwargs):
        """
        初始化增强符号回归算法
        
        参数:
        - test_size: 测试集比例，默认0.2
        - random_state: 随机种子，默认42
        - **pysr_kwargs: PySRRegressor的其他参数
        """
        self.test_size = test_size
        self.random_state = random_state
        
        # 默认PySR参数
        default_params = {
            'model_selection': "best",
            'niterations': 40,
            'binary_operators': ["+", "*"],
            'unary_operators': [
                "cos",
                "exp", 
                "sin",
                "inv(x) = 1/x",
            ],
            'extra_sympy_mappings': {"inv": lambda x: 1 / x},
            'populations': 20,
            'population_size': 35,
            'maxsize': 20,
            'should_optimize_constants': True,
            'optimizer_algorithm': "BFGS",
            'optimizer_nrestarts': 2,
            'weight_simplify': 0.01,
            'weight_optimize': 0.1,
            'parsimony': 0.05,
            'tournament_selection_n': 12,
            'tournament_selection_p': 0.95,
            'verbosity': 1
        }
        
        # 合并用户参数
        default_params.update(pysr_kwargs)
        self.pysr_params = default_params
        
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def _setup_julia_functions(self):
        """设置Julia环境中的多重线性性检测函数"""
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
        
    def fit(self, X, y):
        """
        训练符号回归模型
        
        参数:
        - X: 输入特征矩阵 (n_samples, n_features)
        - y: 目标值向量 (n_samples,)
        
        返回:
        - results: 包含训练和测试结果的字典
        """
        # 划分训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        print(f"数据集划分: 训练集 {len(self.X_train)} 样本, 测试集 {len(self.X_test)} 样本")
        
        # 设置Julia函数
        self._setup_julia_functions()
        
        # 将训练数据传递给Julia环境
        jl.seval("global X_data, y_data")
        jl.X_data = self.X_train
        jl.y_data = self.y_train
        
        # 添加自定义损失函数到参数中
        self.pysr_params['loss_function'] = """
        function enhanced_loss(tree, dataset, options)
            # 评估树得到预测值
            prediction, flag = eval_tree_array(tree, dataset.X, options)
            
            if !flag
                return Inf
            end
            
            # 调用多重线性性优化函数
            return optimize_linear_terms(prediction, dataset.y, tree, dataset, options)
        end
        """
        
        # 创建并训练模型
        self.model = PySRRegressor(**self.pysr_params)
        
        print("开始增强的符号回归搜索（实时多重线性性优化）...")
        self.model.fit(self.X_train, self.y_train)
        
        # 计算性能指标
        results = self._evaluate_performance()
        
        return results
    
    def _evaluate_performance(self):
        """评估模型性能"""
        # 训练集预测
        train_pred = self.model.predict(self.X_train)
        train_mse = np.mean((train_pred - self.y_train) ** 2)
        train_r2 = 1 - (np.sum((self.y_train - train_pred) ** 2) / 
                       np.sum((self.y_train - np.mean(self.y_train)) ** 2))
        
        # 测试集预测
        test_pred = self.model.predict(self.X_test)
        test_mse = np.mean((test_pred - self.y_test) ** 2)
        test_r2 = 1 - (np.sum((self.y_test - test_pred) ** 2) / 
                      np.sum((self.y_test - np.mean(self.y_test)) ** 2))
        
        # 获取最佳方程信息
        best_equation = str(self.model.sympy())
        
        complexity = None
        loss = None
        if hasattr(self.model, 'equations_') and self.model.equations_ is not None:
            best_eq = self.model.equations_.iloc[-1]
            complexity = best_eq['complexity']
            loss = best_eq['loss']
        
        results = {
            'model': self.model,
            'best_equation': best_equation,
            'complexity': complexity,
            'training_loss': loss,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_predictions': train_pred,
            'test_predictions': test_pred,
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test
        }
        
        return results
    
    def predict(self, X):
        """使用训练好的模型进行预测"""
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        return self.model.predict(X)
    
    def print_results(self, results):
        """打印详细结果"""
        print("\n=== 增强符号回归结果 ===")
        print(f"学到的最佳方程: {results['best_equation']}")
        
        if results['complexity'] is not None:
            print(f"方程复杂度: {results['complexity']}")
            print(f"训练损失: {results['training_loss']:.6f}")
        
        print("\n=== 性能评估 ===")
        print(f"训练集 MSE: {results['train_mse']:.6f}")
        print(f"测试集 MSE: {results['test_mse']:.6f}")
        print(f"训练集 R²: {results['train_r2']:.6f}")
        print(f"测试集 R²: {results['test_r2']:.6f}")
        
        # 过拟合检测
        if results['test_mse'] > results['train_mse'] * 1.5:
            print("⚠️  检测到可能的过拟合")
        elif results['test_r2'] > 0.9 and results['train_r2'] > 0.9:
            print("✅ 模型泛化性能良好")

    def save_results(self, results, dataset_name, output_dir="result"):
        """保存结果到本地文件夹"""
        import os
        import json
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存详细结果，转换numpy类型为Python原生类型
        result_data = {
            'dataset_name': dataset_name,
            'best_equation': str(results['best_equation']),
            'complexity': int(results['complexity']) if hasattr(results['complexity'], 'item') else results['complexity'],
            'training_loss': float(results['training_loss']) if hasattr(results['training_loss'], 'item') else results['training_loss'],
            'train_mse': float(results['train_mse']),
            'test_mse': float(results['test_mse']),
            'train_r2': float(results['train_r2']),
            'test_r2': float(results['test_r2'])
        }
        
        # 保存JSON结果
        json_path = os.path.join(output_dir, f"{dataset_name}_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        # 保存预测结果
        pred_path = os.path.join(output_dir, f"{dataset_name}_predictions.txt")
        with open(pred_path, 'w', encoding='utf-8') as f:
            f.write(f"数据集: {dataset_name}\n")
            f.write(f"最佳方程: {results['best_equation']}\n")
            f.write(f"复杂度: {results['complexity']}\n")
            f.write(f"训练集MSE: {results['train_mse']:.6f}\n")
            f.write(f"测试集MSE: {results['test_mse']:.6f}\n")
            f.write(f"训练集R²: {results['train_r2']:.6f}\n")
            f.write(f"测试集R²: {results['test_r2']:.6f}\n\n")
            
            f.write("训练集预测结果:\n")
            for i, (true_val, pred_val) in enumerate(zip(results['y_train'], results['train_predictions'])):
                f.write(f"{i+1:3d}: 真实值={true_val:.6f}, 预测值={pred_val:.6f}, 误差={abs(true_val-pred_val):.6f}\n")
            
            f.write("\n测试集预测结果:\n")
            for i, (true_val, pred_val) in enumerate(zip(results['y_test'], results['test_predictions'])):
                f.write(f"{i+1:3d}: 真实值={true_val:.6f}, 预测值={pred_val:.6f}, 误差={abs(true_val-pred_val):.6f}\n")
        
        # 保存模型方程到单独文件
        equation_path = os.path.join(output_dir, f"{dataset_name}_equation.txt")
        with open(equation_path, 'w', encoding='utf-8') as f:
            f.write(results['best_equation'])
        
        print(f"结果已保存到: {output_dir}/{dataset_name}_*")

# 使用示例
if __name__ == "__main__":
    import pandas as pd
    import os
    
    # 创建结果文件夹
    os.makedirs("result", exist_ok=True)
    
    # 数据集路径
    datasets = [
        ("data/dataset1_5features.txt", "dataset1"),
        ("data/dataset2_5features.txt", "dataset2"), 
        ("data/dataset3_5features.txt", "dataset3")
    ]
    
    # 保存所有数据集的汇总结果
    summary_results = []
    
    for dataset_path, dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"处理数据集: {dataset_name}")
        print(f"{'='*60}")
        
        # 读取数据
        data = pd.read_csv(dataset_path, sep='\t')
        
        # 提取特征和标签
        feature_cols = [col for col in data.columns if col.startswith('C')]
        X = data[feature_cols].values
        y = data['cycle_log'].values
        
        print(f"数据集形状: X={X.shape}, y={y.shape}")
        print(f"特征列: {feature_cols}")
        
        # 创建增强符号回归算法实例
        esr = EnhancedSymbolicRegression(
            test_size=0.2,
            random_state=42,
            niterations=30,
            verbosity=1
        )
        
        # 训练模型
        results = esr.fit(X, y)
        
        # 打印结果
        esr.print_results(results)
        
        # 保存结果
        esr.save_results(results, dataset_name)
        
        # 添加到汇总
        summary_results.append({
            'dataset': dataset_name,
            'equation': results['best_equation'],
            'train_mse': results['train_mse'],
            'test_mse': results['test_mse'],
            'train_r2': results['train_r2'],
            'test_r2': results['test_r2']
        })
        
        print(f"\n数据集 {dataset_name} 处理完成")
    
    # 保存汇总结果
    summary_path = "result/summary_all_datasets.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("所有数据集符号回归结果汇总\n")
        f.write("="*50 + "\n\n")
        
        for result in summary_results:
            f.write(f"数据集: {result['dataset']}\n")
            f.write(f"最佳方程: {result['equation']}\n")
            f.write(f"训练集MSE: {result['train_mse']:.6f}\n")
            f.write(f"测试集MSE: {result['test_mse']:.6f}\n")
            f.write(f"训练集R²: {result['train_r2']:.6f}\n")
            f.write(f"测试集R²: {result['test_r2']:.6f}\n")
            f.write("-"*30 + "\n\n")
    
    print(f"\n所有结果已保存到 result/ 文件夹")
    print(f"汇总结果: {summary_path}")