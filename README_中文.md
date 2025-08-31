[//]: # (Logo:)

<div align="center">

PySR搜索能够优化特定目标的符号表达式。

https://github.com/MilesCranmer/PySR/assets/7593028/c8511a49-b408-488f-8f18-b1749078268f


# PySR: Python和Julia中的高性能符号回归

| **文档** | **论坛** | **论文** | **colab演示** |
|:---:|:---:|:---:|:---:|
|[![Documentation](https://github.com/MilesCranmer/PySR/actions/workflows/docs.yml/badge.svg)](https://ai.damtp.cam.ac.uk/pysr/)|[![Discussions](https://img.shields.io/badge/discussions-github-informational)](https://github.com/MilesCranmer/PySR/discussions)|[![Paper](https://img.shields.io/badge/arXiv-2305.01582-b31b1b)](https://arxiv.org/abs/2305.01582)|[![Colab](https://img.shields.io/badge/colab-notebook-yellow)](https://colab.research.google.com/github/MilesCranmer/PySR/blob/master/examples/pysr_demo.ipynb)|

| **pip** | **conda** | **统计** |
| :---: | :---: | :---: |
|[![PyPI version](https://badge.fury.io/py/pysr.svg)](https://badge.fury.io/py/pysr)|[![Conda Version](https://img.shields.io/conda/vn/conda-forge/pysr.svg)](https://anaconda.org/conda-forge/pysr)|<div align="center">pip: [![Downloads](https://static.pepy.tech/badge/pysr)](https://pypi.org/project/pysr/)<br>conda: [![Anaconda-Server Badge](https://anaconda.org/conda-forge/pysr/badges/downloads.svg)](https://anaconda.org/conda-forge/pysr)</div>|

</div>

如果您觉得PySR有用，请引用论文 [arXiv:2305.01582](https://arxiv.org/abs/2305.01582)。
如果您已经完成了使用PySR的项目，请提交PR来在[研究展示页面](https://ai.damtp.cam.ac.uk/pysr/papers)展示您的工作！

**目录**:

- [为什么选择PySR？](#为什么选择pysr)
- [安装](#安装)
- [快速开始](#快速开始)
- [→ 文档](https://ai.damtp.cam.ac.uk/pysr)
- [贡献者](#贡献者-)

<div align="center">

### 测试状态

| **Linux** | **Windows** | **macOS** |
|---|---|---|
|[![Linux](https://github.com/MilesCranmer/PySR/actions/workflows/CI.yml/badge.svg)](https://github.com/MilesCranmer/PySR/actions/workflows/CI.yml)|[![Windows](https://github.com/MilesCranmer/PySR/actions/workflows/CI_Windows.yml/badge.svg)](https://github.com/MilesCranmer/PySR/actions/workflows/CI_Windows.yml)|[![macOS](https://github.com/MilesCranmer/PySR/actions/workflows/CI_mac.yml/badge.svg)](https://github.com/MilesCranmer/PySR/actions/workflows/CI_mac.yml)|
| **Docker** | **Conda** | **覆盖率** |
|[![Docker](https://github.com/MilesCranmer/PySR/actions/workflows/CI_docker.yml/badge.svg)](https://github.com/MilesCranmer/PySR/actions/workflows/CI_docker.yml)|[![conda-forge](https://github.com/MilesCranmer/PySR/actions/workflows/CI_conda_forge.yml/badge.svg)](https://github.com/MilesCranmer/PySR/actions/workflows/CI_conda_forge.yml)|[![codecov](https://codecov.io/gh/MilesCranmer/PySR/branch/master/graph/badge.svg)](https://codecov.io/gh/MilesCranmer/PySR)|

</div>

## 为什么选择PySR？

PySR是一个用于*符号回归*的开源工具：这是一种机器学习任务，目标是找到一个可解释的符号表达式来优化某些目标。

在数年的开发过程中，PySR从头开始设计，旨在实现：
(1) 尽可能高的性能，
(2) 尽可能高的可配置性，
(3) 易于使用。
PySR与Julia库[SymbolicRegression.jl](https://github.com/MilesCranmer/SymbolicRegression.jl)一起开发，
后者构成了PySR的强大搜索引擎。
这些算法的详细信息在[PySR论文](https://arxiv.org/abs/2305.01582)中有所描述。

符号回归在低维数据集上效果最佳，但也可以通过使用神经网络的"*符号蒸馏*"将这些方法扩展到高维空间，如[2006.11287](https://arxiv.org/abs/2006.11287)中所解释的，我们将其应用于N体问题。在这里，人们本质上使用符号回归将神经网络转换为解析方程。因此，这些工具同时提供了一种明确而强大的方式来解释深度神经网络。

## 安装

### Pip

您可以使用pip安装PySR：

```bash
pip install pysr
```

Julia依赖项将在首次导入时安装。

### Conda

同样，使用conda：

```bash
conda install -c conda-forge pysr
```

<details>
<summary>

### Docker

</summary>

您也可以使用`Dockerfile`在docker容器中安装PySR

1. 克隆此仓库。
2. 在仓库目录中，构建docker容器：
```bash
docker build -t pysr .
```
3. 然后您可以通过IPython执行启动容器：
```bash
docker run -it --rm pysr ipython
```

更多详细信息，请参见[docker部分](#docker)。

</details>

<details>
<summary>

### Apptainer

</summary>

如果您在没有root访问权限的集群上使用PySR，
您可以使用[Apptainer](https://apptainer.org/)构建容器
而不是Docker。`Apptainer.def`文件类似于`Dockerfile`，
可以通过以下方式构建：

```bash
apptainer build --notest pysr.sif Apptainer.def
```

并通过以下方式启动：

```bash
apptainer run pysr.sif
```

</details>

<details>
<summary>

### 故障排除

</summary>

您可能遇到的一个问题可能导致导入时硬崩溃，
并显示类似"`GLIBCXX_...` not found"的消息。这是由于另一个Python依赖项
加载了错误的`libstdc++`库。要解决此问题，您应该修改
`LD_LIBRARY_PATH`变量以引用Julia库。例如，如果Julia版本的`libstdc++.so`位于`$HOME/.julia/juliaup/julia-1.10.0+0.x64.linux.gnu/lib/julia/`
（这在您的系统上可能不同！），您可以添加：

```
export LD_LIBRARY_PATH=$HOME/.julia/juliaup/julia-1.10.0+0.x64.linux.gnu/lib/julia/:$LD_LIBRARY_PATH
```

到您的`.bashrc`或`.zshrc`文件中。

</details>


## 快速开始

您可能希望尝试[这里](https://colab.research.google.com/github/MilesCranmer/PySR/blob/master/examples/pysr_demo.ipynb)的交互式教程，它使用`examples/pysr_demo.ipynb`中的笔记本。

在实践中，我强烈建议使用IPython而不是Jupyter，因为打印效果要好得多。
下面是一个快速演示，您可以将其粘贴到Python运行时中。
首先，让我们导入numpy来生成一些测试数据：

```python
import numpy as np

X = 2 * np.random.randn(100, 5)
y = 2.5382 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 0.5
```

我们创建了一个包含100个数据点的数据集，每个数据点有5个特征。
我们希望建模的关系是 $2.5382 \cos(x_3) + x_0^2 - 0.5$。

现在，让我们创建一个PySR模型并训练它。
PySR的主要接口采用scikit-learn的风格：

```python
from pysr import PySRRegressor

model = PySRRegressor(
    maxsize=20,
    niterations=40,  # < 增加此值以获得更好的结果
    binary_operators=["+", "*"],
    unary_operators=[
        "cos",
        "exp",
        "sin",
        "inv(x) = 1/x",
        # ^ 自定义操作符（julia语法）
    ],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    # ^ 也为SymPy定义操作符
    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    # ^ 自定义损失函数（julia语法）
)
```

这将设置模型进行40次搜索代码迭代，其中包含数十万次突变和方程评估。

让我们在数据集上训练这个模型：

```python
model.fit(X, y)
```

在内部，这启动了一个Julia进程，该进程将进行多线程搜索以拟合数据集的方程。

方程将在训练期间打印，一旦您满意，您可以通过按'q'然后\<enter\>提前退出。

模型拟合后，您可以运行`model.predict(X)`
使用自动选择的表达式查看给定数据集的预测，
或者，例如，`model.predict(X, 3)`查看第3个方程的预测。

您可以运行：

```python
print(model)
```

打印学习到的方程：

```python
PySRRegressor.equations_ = [
	   pick     score                                           equation       loss  complexity
	0        0.000000                                          4.4324794  42.354317           1
	1        1.255691                                          (x0 * x0)   3.437307           3
	2        0.011629                          ((x0 * x0) + -0.28087974)   3.358285           5
	3        0.897855                              ((x0 * x0) + cos(x3))   1.368308           6
	4        0.857018                ((x0 * x0) + (cos(x3) * 2.4566472))   0.246483           8
	5  >>>>       inf  (((cos(x3) + -0.19699033) * 2.5382123) + (x0 *...   0.000000          10
]
```

`pick`列中的箭头表示您的`model_selection`策略当前选择用于预测的方程。
（您也可以在`.fit(X, y)`之后更改`model_selection`。）

`model.equations_`是一个包含所有方程的pandas DataFrame，包括可调用格式
（`lambda_format`）、
SymPy格式（`sympy_format` - 您也可以通过`model.sympy()`获得）、甚至JAX和PyTorch格式
（两者都是可微分的 - 您可以通过`model.jax()`和`model.pytorch()`获得）。

请注意，`PySRRegressor`存储最后一次搜索的状态，如果您设置了`warm_start=True`，下次调用`.fit()`时将从上次停止的地方重新开始。
如果对搜索参数进行了重大更改（如更改操作符），这会导致问题。您可以运行`model.reset()`来重置状态。

您会注意到PySR将保存两个文件：
`hall_of_fame...csv`和`hall_of_fame...pkl`。
csv文件是方程及其损失的列表，pkl文件是模型的保存状态。
您可以从pkl文件加载模型：

```python
model = PySRRegressor.from_file("hall_of_fame.2022-08-10_100832.281.pkl")
```

还有几个其他有用的功能，如去噪（例如，`denoise=True`）、
特征选择（例如，`select_k_features=3`）。
有关这些和其他功能的示例，请参见[示例页面](https://ai.damtp.cam.ac.uk/pysr/examples)。
有关更多选项的详细了解，请参见[选项页面](https://ai.damtp.cam.ac.uk/pysr/options)。
您也可以在[此页面](https://ai.damtp.cam.ac.uk/pysr/api)查看完整的API。
还有[此页面](https://ai.damtp.cam.ac.uk/pysr/tuning)上的PySR调优技巧。

### 详细示例

以下代码尽可能多地使用了PySR功能。
请注意，这只是功能演示，您不应按原样使用此示例。
有关每个参数的详细信息，请查看[API页面](https://ai.damtp.cam.ac.uk/pysr/api/)。

```python
model = PySRRegressor(
    populations=8,
    # ^ 假设我们有4个核心，这意味着每个核心2个种群，所以总有一个在运行。
    population_size=50,
    # ^ 稍大的种群，以获得更大的多样性。
    ncycles_per_iteration=500,
    # ^ 迁移之间的代数。
    niterations=10000000,  # 永远运行
    early_stop_condition=(
        "stop_if(loss, complexity) = loss < 1e-6 && complexity < 10"
        # 如果我们找到一个好且简单的方程，就提前停止
    ),
    timeout_in_seconds=60 * 60 * 24,
    # ^ 或者，24小时后停止。
    maxsize=50,
    # ^ 允许更大的复杂性。
    maxdepth=10,
    # ^ 但是，避免深度嵌套。
    binary_operators=["*", "+", "-", "/"],
    unary_operators=["square", "cube", "exp", "cos2(x)=cos(x)^2"],
    constraints={
        "/": (-1, 9),
        "square": 9,
        "cube": 9,
        "exp": 9,
    },
    # ^ 限制每个参数内的复杂性。
    # "inv": (-1, 9) 表示分子没有约束，
    # 但分母的最大复杂性为9。
    # "exp": 9 简单地表示 `exp` 只能有
    # 复杂性为9的表达式作为输入。
    nested_constraints={
        "square": {"square": 1, "cube": 1, "exp": 0},
        "cube": {"square": 1, "cube": 1, "exp": 0},
        "exp": {"square": 1, "cube": 1, "exp": 0},
    },
    # ^ 操作符的嵌套约束。例如，
    # "square(exp(x))" 不被允许，因为 "square": {"exp": 0}。
    complexity_of_operators={"/": 2, "exp": 3},
    # ^ 特定操作符的自定义复杂性。
    complexity_of_constants=2,
    # ^ 比变量更多地惩罚常数
    select_k_features=4,
    # ^ 只在4个最重要的特征上训练
    progress=True,
    # ^ 如果打印到文件，可以设置为false。
    weight_randomize=0.1,
    # ^ 更频繁地随机化树
    cluster_manager=None,
    # ^ 可以设置为，例如，"slurm"，以运行slurm
    # 集群。只需从头节点启动一个脚本。
    precision=64,
    # ^ 更高精度的计算。
    warm_start=True,
    # ^ 从上次停止的地方开始。
    turbo=True,
    # ^ 更快的评估（实验性）
    extra_sympy_mappings={"cos2": lambda x: sympy.cos(x)**2},
    # extra_torch_mappings={sympy.cos: torch.cos},
    # ^ 不需要，因为cos已经定义，但这是
    # 您定义自定义torch操作符的方式。
    # extra_jax_mappings={sympy.cos: "jnp.cos"},
    # ^ 对于JAX，传递一个字符串。
)
```

### Docker

您也可以在Docker中测试PySR，无需
在本地安装，通过在此仓库的根目录中运行以下命令：

```bash
docker build -t pysr .
```

这为您的系统架构构建了一个名为`pysr`的镜像，
它也包含IPython。您可以选择特定版本的Python和Julia：

```bash
docker build -t pysr --build-arg JLVERSION=1.10.0 --build-arg PYVERSION=3.11.6 .
```

然后您可以使用此dockerfile运行：

```bash
docker run -it --rm -v "$PWD:/data" pysr ipython
```

这将把当前目录链接到容器的`/data`目录
然后启动ipython。

如果您在为系统架构构建时遇到问题，
您可以通过在`build`和`run`命令之前包含`--platform linux/amd64`来模拟另一个架构。

<div align="center">

### 贡献者 ✨

</div>

我们热切欢迎新的贡献者！查看我们的贡献者[指南](https://github.com/MilesCranmer/PySR/blob/master/CONTRIBUTORS.md)以获取技巧 🚀。
如果您有新功能的想法，请不要犹豫在[问题](https://github.com/MilesCranmer/PySR/issues)或[讨论](https://github.com/MilesCranmer/PySR/discussions)页面分享。

<!-- 贡献者列表与英文版相同 -->