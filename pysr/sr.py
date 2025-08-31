"""Define the PySRRegressor scikit-learn interface."""

from __future__ import annotations

import copy
import logging
import os
import pickle as pkl
import re
import sys
import tempfile
import warnings
from collections.abc import Callable
from dataclasses import dataclass, fields
from io import StringIO
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Literal, Tuple, Union, cast

import numpy as np
import pandas as pd
from numpy import ndarray
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin
from sklearn.utils import check_array, check_consistent_length, check_random_state
from sklearn.utils.validation import _check_feature_names_in  # type: ignore
from sklearn.utils.validation import check_is_fitted

from .denoising import denoise, multi_denoise
from .deprecated import DEPRECATED_KWARGS
from .export_latex import (
    sympy2latex,
    sympy2latextable,
    sympy2multilatextable,
    with_preamble,
)
from .export_sympy import assert_valid_sympy_symbol
from .expression_specs import (
    AbstractExpressionSpec,
    ExpressionSpec,
    ParametricExpressionSpec,
    parametric_expression_deprecation_warning,
)
from .feature_selection import run_feature_selection
from .julia_extensions import load_required_packages
from .julia_helpers import (
    _escape_filename,
    _load_cluster_manager,
    jl_array,
    jl_deserialize,
    jl_is_function,
    jl_serialize,
)
from .julia_import import AnyValue, SymbolicRegression, VectorValue, jl
from .logger_specs import AbstractLoggerSpec
from .utils import (
    ArrayLike,
    PathLike,
    _preprocess_julia_floats,
    _safe_check_feature_names_in,
    _subscriptify,
    _suggest_keywords,
)

try:
    from sklearn.utils.validation import validate_data

    OLD_SKLEARN = False
except ImportError:
    OLD_SKLEARN = True

try:
    from typing import List
except ImportError:
    from typing_extensions import List


ALREADY_RAN = False

pysr_logger = logging.getLogger(__name__)


def _process_constraints(
    binary_operators: list[str],
    unary_operators: list,
    constraints: dict[str, int | tuple[int, int]],
) -> dict[str, int | tuple[int, int]]:
    constraints = constraints.copy()
    for op in unary_operators:
        if op not in constraints:
            constraints[op] = -1
    for op in binary_operators:
        if op not in constraints:
            if op in ["^", "pow"]:
                # Warn user that they should set up constraints
                warnings.warn(
                    "You are using the `^` operator, but have not set up `constraints` for it. "
                    "This may lead to overly complex expressions. "
                    "One typical constraint is to use `constraints={..., '^': (-1, 1)}`, which "
                    "will allow arbitrary-complexity base (-1) but only powers such as "
                    "a constant or variable (1). "
                    "For more tips, please see https://ai.damtp.cam.ac.uk/pysr/tuning/"
                )
            constraints[op] = (-1, -1)

        constraint_tuple = cast(Tuple[int, int], constraints[op])
        if op in ["plus", "sub", "+", "-"]:
            if constraint_tuple[0] != constraint_tuple[1]:
                raise NotImplementedError(
                    "You need equal constraints on both sides for - and +, "
                    "due to simplification strategies."
                )
        elif op in ["mult", "*"]:
            # Make sure the complex expression is in the left side.
            if constraint_tuple[0] == -1:
                continue
            if constraint_tuple[1] == -1 or constraint_tuple[0] < constraint_tuple[1]:
                constraints[op] = (constraint_tuple[1], constraint_tuple[0])
    return constraints


def _maybe_create_inline_operators(
    binary_operators: list[str],
    unary_operators: list[str],
    extra_sympy_mappings: dict[str, Callable] | None,
    expression_spec: AbstractExpressionSpec,
) -> tuple[list[str], list[str]]:
    binary_operators = binary_operators.copy()
    unary_operators = unary_operators.copy()
    for op_list in [binary_operators, unary_operators]:
        for i, op in enumerate(op_list):
            is_user_defined_operator = "(" in op

            if is_user_defined_operator:
                jl.seval(op)
                # Cut off from the first non-alphanumeric char:
                first_non_char = [j for j, char in enumerate(op) if char == "("][0]
                function_name = op[:first_non_char]
                # Assert that function_name only contains
                # alphabetical characters, numbers,
                # and underscores:
                if not re.match(r"^[a-zA-Z0-9_]+$", function_name):
                    raise ValueError(
                        f"Invalid function name {function_name}. "
                        "Only alphanumeric characters, numbers, "
                        "and underscores are allowed."
                    )
                missing_sympy_mapping = (
                    extra_sympy_mappings is None
                    or function_name not in extra_sympy_mappings
                )
                if missing_sympy_mapping and expression_spec.supports_sympy:
                    raise ValueError(
                        f"Custom function {function_name} is not defined in `extra_sympy_mappings`. "
                        "You can define it with, "
                        "e.g., `model.set_params(extra_sympy_mappings={'inv': lambda x: 1/x})`, where "
                        "`lambda x: 1/x` is a valid SymPy function defining the operator. "
                        "You can also define these at initialization time."
                    )
                op_list[i] = function_name
    return binary_operators, unary_operators


def _check_assertions(
    X,
    use_custom_variable_names,
    variable_names,
    complexity_of_variables,
    weights,
    y,
    X_units,
    y_units,
):
    # 在错误发生之前检查潜在的错误
    
    # 检查X必须是二维数组（样本数×特征数）
    assert len(X.shape) == 2
    
    # 检查y必须是一维或二维数组
    assert len(y.shape) in [1, 2]
    
    # 检查X和y的样本数量必须一致
    assert X.shape[0] == y.shape[0]
    
    # 如果提供了权重，检查权重的相关属性
    if weights is not None:
        # 权重的形状必须与y一致
        assert weights.shape == y.shape
        # 权重的样本数必须与X一致
        assert X.shape[0] == weights.shape[0]
    
    # 如果使用自定义变量名，进行相关检查
    if use_custom_variable_names:
        # 变量名的数量必须与特征数一致
        assert len(variable_names) == X.shape[1]
        # 检查变量名不能是函数名
        for var_name in variable_names:
            # 检查变量名是否只包含字母数字字符
            if not re.match(r"^[₀₁₂₃₄₅₆₇₈₉a-zA-Z0-9_]+$", var_name):
                raise ValueError(
                    f"无效的变量名 {var_name}。"
                    "只允许使用字母数字字符、数字和下划线。"
                )
            # 验证变量名是否为有效的SymPy符号
            assert_valid_sympy_symbol(var_name)
    
    # 如果complexity_of_variables是列表，检查其长度是否与特征数一致
    if (
        isinstance(complexity_of_variables, list)
        and len(complexity_of_variables) != X.shape[1]
    ):
        raise ValueError(
            "complexity_of_variables中的元素数量必须等于X中的特征数量。"
        )
    
    # 如果提供了X_units，检查其长度是否与特征数一致
    if X_units is not None and len(X_units) != X.shape[1]:
        raise ValueError(
            "X_units中的单位数量必须等于X中的特征数量。"
        )
    
    # 如果提供了y_units，进行相关检查
    if y_units is not None:
        # 初始化检查标志
        good_y_units = False
        # 如果y_units是列表形式
        if isinstance(y_units, list):
            # 如果y是一维的，检查y_units是否只有一个元素
            if len(y.shape) == 1:
                good_y_units = len(y_units) == 1
            else:
                # 如果y是二维的，检查y_units的长度是否与y的列数一致
                good_y_units = len(y_units) == y.shape[1]
        else:
            # 如果y_units不是列表，检查y是否是一维或者只有一列
            good_y_units = len(y.shape) == 1 or y.shape[1] == 1

        # 如果y_units检查不通过，抛出错误
        if not good_y_units:
            raise ValueError(
                "y_units中的单位数量必须等于y中的输出特征数量。"
            )


def _validate_export_mappings(extra_jax_mappings, extra_torch_mappings):
    # It is expected extra_jax/torch_mappings will be updated after fit.
    # Thus, validation is performed here instead of in _validate_init_params
    if extra_jax_mappings is not None:
        for value in extra_jax_mappings.values():
            if not isinstance(value, str):
                raise ValueError(
                    "extra_jax_mappings must have keys that are strings! "
                    "e.g., {sympy.sqrt: 'jnp.sqrt'}."
                )
    if extra_torch_mappings is not None:
        for value in extra_torch_mappings.values():
            if not callable(value):
                raise ValueError(
                    "extra_torch_mappings must be callable functions! "
                    "e.g., {sympy.sqrt: torch.sqrt}."
                )


# Class validation constants
VALID_OPTIMIZER_ALGORITHMS = ["BFGS", "NelderMead"]


@dataclass
class _DynamicallySetParams:
    """Defines some parameters that are set at runtime."""

    binary_operators: list[str]
    unary_operators: list[str]
    maxdepth: int
    constraints: dict[str, int | tuple[int, int]]
    batch_size: int
    update_verbosity: int
    progress: bool
    warmup_maxsize_by: float


class PySRRegressor(MultiOutputMixin, RegressorMixin, BaseEstimator):
    """
    High-performance symbolic regression algorithm.

    This is the scikit-learn interface for SymbolicRegression.jl.
    This model will automatically search for equations which fit
    a given dataset subject to a particular loss and set of
    constraints.

    Most default parameters have been tuned over several example equations,
    but you should adjust `niterations`, `binary_operators`, `unary_operators`
    to your requirements. You can view more detailed explanations of the options
    on the [options page](https://ai.damtp.cam.ac.uk/pysr/options) of the
    documentation.

    Parameters
    ----------
    model_selection : str
        Model selection criterion when selecting a final expression from
        the list of best expression at each complexity.
        Can be `'accuracy'`, `'best'`, or `'score'`. Default is `'best'`.
        `'accuracy'` selects the candidate model with the lowest loss
        (highest accuracy).
        `'score'` selects the candidate model with the highest score.
        Score is defined as the negated derivative of the log-loss with
        respect to complexity - if an expression has a much better
        loss at a slightly higher complexity, it is preferred.
        `'best'` selects the candidate model with the highest score
        among expressions with a loss better than at least 1.5x the
        most accurate model.
    binary_operators : list[str]
        List of strings for binary operators used in the search.
        See the [operators page](https://ai.damtp.cam.ac.uk/pysr/operators/)
        for more details.
        Default is `["+", "-", "*", "/"]`.
    unary_operators : list[str]
        Operators which only take a single scalar as input.
        For example, `"cos"` or `"exp"`.
        Default is `None`.
    expression_spec : AbstractExpressionSpec
        The type of expression to search for. By default,
        this is just `ExpressionSpec()`. You can also use
        `TemplateExpressionSpec(...)` which allows you to specify
        a custom template for the expressions.
        Default is `ExpressionSpec()`.
    niterations : int
        Number of iterations of the algorithm to run. The best
        equations are printed and migrate between populations at the
        end of each iteration.
        Default is `100`.
    populations : int
        Number of populations running.
        Default is `31`.
    population_size : int
        Number of individuals in each population.
        Default is `27`.
    max_evals : int
        Limits the total number of evaluations of expressions to
        this number.  Default is `None`.
    maxsize : int
        Max complexity of an equation.  Default is `30`.
    maxdepth : int
        Max depth of an equation. You can use both `maxsize` and
        `maxdepth`. `maxdepth` is by default not used.
        Default is `None`.
    warmup_maxsize_by : float
        Whether to slowly increase max size from a small number up to
        the maxsize (if greater than 0).  If greater than 0, says the
        fraction of training time at which the current maxsize will
        reach the user-passed maxsize.
        Default is `0.0`.
    timeout_in_seconds : float
        Make the search return early once this many seconds have passed.
        Default is `None`.
    constraints : dict[str, int | tuple[int,int]]
        Dictionary of int (unary) or 2-tuples (binary), this enforces
        maxsize constraints on the individual arguments of operators.
        E.g., `'pow': (-1, 1)` says that power laws can have any
        complexity left argument, but only 1 complexity in the right
        argument. Use this to force more interpretable solutions.
        Default is `None`.
    nested_constraints : dict[str, dict]
        Specifies how many times a combination of operators can be
        nested. For example, `{"sin": {"cos": 0}}, "cos": {"cos": 2}}`
        specifies that `cos` may never appear within a `sin`, but `sin`
        can be nested with itself an unlimited number of times. The
        second term specifies that `cos` can be nested up to 2 times
        within a `cos`, so that `cos(cos(cos(x)))` is allowed
        (as well as any combination of `+` or `-` within it), but
        `cos(cos(cos(cos(x))))` is not allowed. When an operator is not
        specified, it is assumed that it can be nested an unlimited
        number of times. This requires that there is no operator which
        is used both in the unary operators and the binary operators
        (e.g., `-` could be both subtract, and negation). For binary
        operators, you only need to provide a single number: both
        arguments are treated the same way, and the max of each
        argument is constrained.
        Default is `None`.
    elementwise_loss : str
        String of Julia code specifying an elementwise loss function.
        Can either be a loss from LossFunctions.jl, or your own loss
        written as a function. Examples of custom written losses include:
        `myloss(x, y) = abs(x-y)` for non-weighted, or
        `myloss(x, y, w) = w*abs(x-y)` for weighted.
        The included losses include:
        Regression: `LPDistLoss{P}()`, `L1DistLoss()`,
        `L2DistLoss()` (mean square), `LogitDistLoss()`,
        `HuberLoss(d)`, `L1EpsilonInsLoss(ϵ)`, `L2EpsilonInsLoss(ϵ)`,
        `PeriodicLoss(c)`, `QuantileLoss(τ)`.
        Classification: `ZeroOneLoss()`, `PerceptronLoss()`,
        `L1HingeLoss()`, `SmoothedL1HingeLoss(γ)`,
        `ModifiedHuberLoss()`, `L2MarginLoss()`, `ExpLoss()`,
        `SigmoidLoss()`, `DWDMarginLoss(q)`.
        Default is `"L2DistLoss()"`.
    loss_function : str
        Alternatively, you can specify the full objective function as
        a snippet of Julia code, including any sort of custom evaluation
        (including symbolic manipulations beforehand), and any sort
        of loss function or regularizations. The default `loss_function`
        used in SymbolicRegression.jl is roughly equal to:
        ```julia
        function eval_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
            prediction, flag = eval_tree_array(tree, dataset.X, options)
            if !flag
                return L(Inf)
            end
            return sum((prediction .- dataset.y) .^ 2) / dataset.n
        end
        ```
        where the example elementwise loss is mean-squared error.
        You may pass a function with the same arguments as this (note
        that the name of the function doesn't matter). Here,
        both `prediction` and `dataset.y` are 1D arrays of length `dataset.n`.
        Default is `None`.
    loss_function_expression : str
        Similar to `loss_function`, but takes as input the full
        expression object as the first argument, rather than
        the innermost `AbstractExpressionNode`. This is useful
        for specifying custom loss functions on `TemplateExpressionSpec`.
        Default is `None`.
    loss_scale : Literal["log", "linear"]
        Determines how loss values are scaled when computing scores.
        "log" (default) uses logarithmic scaling of loss ratios; this mode
        requires non-negative loss values and is ideal for traditional
        loss functions that are always non-negative.
        "linear" uses direct
        differences between losses; this mode handles any loss values
        (including negative) and is useful for custom loss functions,
        especially those based on likelihoods.
        Default is "log".
    complexity_of_operators : dict[str, int | float]
        If you would like to use a complexity other than 1 for an
        operator, specify the complexity here. For example,
        `{"sin": 2, "+": 1}` would give a complexity of 2 for each use
        of the `sin` operator, and a complexity of 1 for each use of
        the `+` operator (which is the default). You may specify real
        numbers for a complexity, and the total complexity of a tree
        will be rounded to the nearest integer after computing.
        Default is `None`.
    complexity_of_constants : int | float
        Complexity of constants. Default is `1`.
    complexity_of_variables : int | float | list[int | float]
        Global complexity of variables. To set different complexities for
        different variables, pass a list of complexities to the `fit` method
        with keyword `complexity_of_variables`. You cannot use both.
        Default is `1`.
    complexity_mapping : str
        Alternatively, you can pass a function (a string of Julia code) that
        takes the expression as input and returns the complexity. Make sure that
        this operates on `AbstractExpression` (and unpacks to `AbstractExpressionNode`),
        and returns an integer.
        Default is `None`.
    parsimony : float
        Multiplicative factor for how much to punish complexity.
        Default is `0.0`.
    dimensional_constraint_penalty : float
        Additive penalty for if dimensional analysis of an expression fails.
        By default, this is `1000.0`.
    dimensionless_constants_only : bool
        Whether to only search for dimensionless constants, if using units.
        Default is `False`.
    use_frequency : bool
        Whether to measure the frequency of complexities, and use that
        instead of parsimony to explore equation space. Will naturally
        find equations of all complexities.
        Default is `True`.
    use_frequency_in_tournament : bool
        Whether to use the frequency mentioned above in the tournament,
        rather than just the simulated annealing.
        Default is `True`.
    adaptive_parsimony_scaling : float
        If the adaptive parsimony strategy (`use_frequency` and
        `use_frequency_in_tournament`), this is how much to (exponentially)
        weight the contribution. If you find that the search is only optimizing
        the most complex expressions while the simpler expressions remain stagnant,
        you should increase this value.
        Default is `1040.0`.
    alpha : float
        Initial temperature for simulated annealing
        (requires `annealing` to be `True`).
        Default is `3.17`.
    annealing : bool
        Whether to use annealing.  Default is `False`.
    early_stop_condition : float | str
        Stop the search early if this loss is reached. You may also
        pass a string containing a Julia function which
        takes a loss and complexity as input, for example:
        `"f(loss, complexity) = (loss < 0.1) && (complexity < 10)"`.
        Default is `None`.
    ncycles_per_iteration : int
        Number of total mutations to run, per 10 samples of the
        population, per iteration.
        Default is `380`.
    fraction_replaced : float
        How much of population to replace with migrating equations from
        other populations.
        Default is `0.00036`.
    fraction_replaced_hof : float
        How much of population to replace with migrating equations from
        hall of fame. Default is `0.0614`.
    weight_add_node : float
        Relative likelihood for mutation to add a node.
        Default is `2.47`.
    weight_insert_node : float
        Relative likelihood for mutation to insert a node.
        Default is `0.0112`.
    weight_delete_node : float
        Relative likelihood for mutation to delete a node.
        Default is `0.870`.
    weight_do_nothing : float
        Relative likelihood for mutation to leave the individual.
        Default is `0.273`.
    weight_mutate_constant : float
        Relative likelihood for mutation to change the constant slightly
        in a random direction.
        Default is `0.0346`.
    weight_mutate_operator : float
        Relative likelihood for mutation to swap an operator.
        Default is `0.293`.
    weight_swap_operands : float
        Relative likehood for swapping operands in binary operators.
        Default is `0.198`.
    weight_rotate_tree : float
        How often to perform a tree rotation at a random node.
        Default is `4.26`.
    weight_randomize : float
        Relative likelihood for mutation to completely delete and then
        randomly generate the equation
        Default is `0.000502`.
    weight_simplify : float
        Relative likelihood for mutation to simplify constant parts by evaluation
        Default is `0.00209`.
    weight_optimize: float
        Constant optimization can also be performed as a mutation, in addition to
        the normal strategy controlled by `optimize_probability` which happens
        every iteration. Using it as a mutation is useful if you want to use
        a large `ncycles_periteration`, and may not optimize very often.
        Default is `0.0`.
    crossover_probability : float
        Absolute probability of crossover-type genetic operation, instead of a mutation.
        Default is `0.0259`.
    skip_mutation_failures : bool
        Whether to skip mutation and crossover failures, rather than
        simply re-sampling the current member.
        Default is `True`.
    migration : bool
        Whether to migrate.  Default is `True`.
    hof_migration : bool
        Whether to have the hall of fame migrate.  Default is `True`.
    topn : int
        How many top individuals migrate from each population.
        Default is `12`.
    should_simplify : bool
        Whether to use algebraic simplification in the search. Note that only
        a few simple rules are implemented. Default is `True`.
    should_optimize_constants : bool
        Whether to numerically optimize constants (Nelder-Mead/Newton)
        at the end of each iteration. Default is `True`.
    optimizer_algorithm : str
        Optimization scheme to use for optimizing constants. Can currently
        be `NelderMead` or `BFGS`.
        Default is `"BFGS"`.
    optimizer_nrestarts : int
        Number of time to restart the constants optimization process with
        different initial conditions.
        Default is `2`.
    optimizer_f_calls_limit : int
        How many function calls to allow during optimization.
        Default is `10_000`.
    optimize_probability : float
        Probability of optimizing the constants during a single iteration of
        the evolutionary algorithm.
        Default is `0.14`.
    optimizer_iterations : int
        Number of iterations that the constants optimizer can take.
        Default is `8`.
    perturbation_factor : float
        Constants are perturbed by a max factor of
        (perturbation_factor*T + 1). Either multiplied by this or
        divided by this.
        Default is `0.129`.
    probability_negate_constant : float
        Probability of negating a constant in the equation when mutating it.
        Default is `0.00743`.
    tournament_selection_n : int
        Number of expressions to consider in each tournament.
        Default is `15`.
    tournament_selection_p : float
        Probability of selecting the best expression in each
        tournament. The probability will decay as p*(1-p)^n for other
        expressions, sorted by loss.
        Default is `0.982`.
    parallelism: Literal["serial", "multithreading", "multiprocessing"] | None
        Parallelism to use for the search. Can be `"serial"`, `"multithreading"`, or `"multiprocessing"`.
        Default is `"multithreading"`.
    procs: int | None
        Number of processes to use for parallelism. If `None`, defaults to `cpu_count()`.
        Default is `None`.
    cluster_manager : str
        For distributed computing, this sets the job queue system. Set
        to one of "slurm", "pbs", "lsf", "sge", "qrsh", "scyld", or
        "htc". If set to one of these, PySR will run in distributed
        mode, and use `procs` to figure out how many processes to launch.
        Default is `None`.
    heap_size_hint_in_bytes : int
        For multiprocessing, this sets the `--heap-size-hint` parameter
        for new Julia processes. This can be configured when using
        multi-node distributed compute, to give a hint to each process
        about how much memory they can use before aggressive garbage
        collection.
    batching : bool
        Whether to compare population members on small batches during
        evolution. Still uses full dataset for comparing against hall
        of fame. Default is `False`.
    batch_size : int
        The amount of data to use if doing batching. Default is `50`.
    fast_cycle : bool
        Batch over population subsamples. This is a slightly different
        algorithm than regularized evolution, but does cycles 15%
        faster. May be algorithmically less efficient.
        Default is `False`.
    turbo: bool
        (Experimental) Whether to use LoopVectorization.jl to speed up the
        search evaluation. Certain operators may not be supported.
        Does not support 16-bit precision floats.
        Default is `False`.
    bumper: bool
        (Experimental) Whether to use Bumper.jl to speed up the search
        evaluation. Does not support 16-bit precision floats.
        Default is `False`.
    precision : int
        What precision to use for the data. By default this is `32`
        (float32), but you can select `64` or `16` as well, giving
        you 64 or 16 bits of floating point precision, respectively.
        If you pass complex data, the corresponding complex precision
        will be used (i.e., `64` for complex128, `32` for complex64).
        Default is `32`.
    autodiff_backend : Literal["Zygote"] | None
        Which backend to use for automatic differentiation during constant
        optimization. Currently only `"Zygote"` is supported. The default,
        `None`, uses forward-mode or finite difference.
        Default is `None`.
    random_state : int, Numpy RandomState instance or None
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.
        Default is `None`.
    deterministic : bool
        Make a PySR search give the same result every run.
        To use this, you must turn off parallelism
        (with `parallelism="serial"`),
        and set `random_state` to a fixed seed.
        Default is `False`.
    warm_start : bool
        Tells fit to continue from where the last call to fit finished.
        If false, each call to fit will be fresh, overwriting previous results.
        Default is `False`.
    verbosity : int
        What verbosity level to use. 0 means minimal print statements.
        Default is `1`.
    update_verbosity : int
        What verbosity level to use for package updates.
        Will take value of `verbosity` if not given.
        Default is `None`.
    print_precision : int
        How many significant digits to print for floats. Default is `5`.
    progress : bool
        Whether to use a progress bar instead of printing to stdout.
        Default is `True`.
    logger_spec: AbstractLoggerSpec | None
        Logger specification for the Julia backend. See, for example,
        `TensorBoardLoggerSpec`.
        Default is `None`.
    input_stream : str
        The stream to read user input from. By default, this is `"stdin"`.
        If you encounter issues with reading from `stdin`, like a hang,
        you can simply pass `"devnull"` to this argument. You can also
        reference an arbitrary Julia object in the `Main` namespace.
        Default is `"stdin"`.
    run_id : str
        A unique identifier for the run. Will be generated using the
        current date and time if not provided.
        Default is `None`.
    output_directory : str
        The base directory to save output files to. Files
        will be saved in a subdirectory according to the run ID.
        Will be set to `outputs/` if not provided.
        Default is `None`.
    temp_equation_file : bool
        Whether to put the hall of fame file in the temp directory.
        Deletion is then controlled with the `delete_tempfiles`
        parameter.
        Default is `False`.
    tempdir : str
        directory for the temporary files. Default is `None`.
    delete_tempfiles : bool
        Whether to delete the temporary files after finishing.
        Default is `True`.
    update: bool
        Whether to automatically update Julia packages when `fit` is called.
        You should make sure that PySR is up-to-date itself first, as
        the packaged Julia packages may not necessarily include all
        updated dependencies.
        Default is `False`.
    output_jax_format : bool
        Whether to create a 'jax_format' column in the output,
        containing jax-callable functions and the default parameters in
        a jax array.
        Default is `False`.
    output_torch_format : bool
        Whether to create a 'torch_format' column in the output,
        containing a torch module with trainable parameters.
        Default is `False`.
    extra_sympy_mappings : dict[str, Callable]
        Provides mappings between custom `binary_operators` or
        `unary_operators` defined in julia strings, to those same
        operators defined in sympy.
        E.G if `unary_operators=["inv(x)=1/x"]`, then for the fitted
        model to be export to sympy, `extra_sympy_mappings`
        would be `{"inv": lambda x: 1/x}`.
        Default is `None`.
    extra_jax_mappings : dict[Callable, str]
        Similar to `extra_sympy_mappings` but for model export
        to jax. The dictionary maps sympy functions to jax functions.
        For example: `extra_jax_mappings={sympy.sin: "jnp.sin"}` maps
        the `sympy.sin` function to the equivalent jax expression `jnp.sin`.
        Default is `None`.
    extra_torch_mappings : dict[Callable, Callable]
        The same as `extra_jax_mappings` but for model export
        to pytorch. Note that the dictionary keys should be callable
        pytorch expressions.
        For example: `extra_torch_mappings={sympy.sin: torch.sin}`.
        Default is `None`.
    denoise : bool
        Whether to use a Gaussian Process to denoise the data before
        inputting to PySR. Can help PySR fit noisy data.
        Default is `False`.
    select_k_features : int
        Whether to run feature selection in Python using random forests,
        before passing to the symbolic regression code. None means no
        feature selection; an int means select that many features.
        Default is `None`.
    **kwargs : dict
        Supports deprecated keyword arguments. Other arguments will
        result in an error.
    Attributes
    ----------
    equations_ : pandas.DataFrame | list[pandas.DataFrame]
        Processed DataFrame containing the results of model fitting.
    n_features_in_ : int
        Number of features seen during :term:`fit`.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
    display_feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Pretty names of features, used only during printing.
    X_units_ : list[str] of length n_features
        Units of each variable in the training dataset, `X`.
    y_units_ : str | list[str] of length n_out
        Units of each variable in the training dataset, `y`.
    nout_ : int
        Number of output dimensions.
    selection_mask_ : ndarray of shape (`n_features_in_`,)
        Mask of which features of `X` to use when `select_k_features` is set.
    tempdir_ : Path | None
        Path to the temporary equations directory.
    julia_state_stream_ : ndarray
        The serialized state for the julia SymbolicRegression.jl backend (after fitting),
        stored as an array of uint8, produced by Julia's Serialization.serialize function.
    julia_options_stream_ : ndarray
        The serialized julia options, stored as an array of uint8,
    logger_ : AnyValue | None
        The logger instance used for this fit, if any.
    expression_spec_ : AbstractExpressionSpec
        The expression specification used for this fit. This is equal to
        `self.expression_spec` if provided, or `ExpressionSpec()` otherwise.
    equation_file_contents_ : list[pandas.DataFrame]
        Contents of the equation file output by the Julia backend.
    show_pickle_warnings_ : bool
        Whether to show warnings about what attributes can be pickled.

    Examples
    --------
    ```python
    >>> import numpy as np
    >>> from pysr import PySRRegressor
    >>> randstate = np.random.RandomState(0)
    >>> X = 2 * randstate.randn(100, 5)
    >>> # y = 2.5382 * cos(x_3) + x_0 - 0.5
    >>> y = 2.5382 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 0.5
    >>> model = PySRRegressor(
    ...     niterations=40,
    ...     binary_operators=["+", "*"],
    ...     unary_operators=[
    ...         "cos",
    ...         "exp",
    ...         "sin",
    ...         "inv(x) = 1/x",  # Custom operator (julia syntax)
    ...     ],
    ...     model_selection="best",
    ...     elementwise_loss="loss(x, y) = (x - y)^2",  # Custom loss function (julia syntax)
    ... )
    >>> model.fit(X, y)
    >>> model
    PySRRegressor.equations_ = [
    0         0.000000                                          3.8552167  3.360272e+01           1
    1         1.189847                                          (x0 * x0)  3.110905e+00           3
    2         0.010626                          ((x0 * x0) + -0.25573406)  3.045491e+00           5
    3         0.896632                              (cos(x3) + (x0 * x0))  1.242382e+00           6
    4         0.811362                ((x0 * x0) + (cos(x3) * 2.4384754))  2.451971e-01           8
    5  >>>>  13.733371          (((cos(x3) * 2.5382) + (x0 * x0)) + -0.5)  2.889755e-13          10
    6         0.194695  ((x0 * x0) + (((cos(x3) + -0.063180044) * 2.53...  1.957723e-13          12
    7         0.006988  ((x0 * x0) + (((cos(x3) + -0.32505524) * 1.538...  1.944089e-13          13
    8         0.000955  (((((x0 * x0) + cos(x3)) + -0.8251649) + (cos(...  1.940381e-13          15
    ]
    >>> model.score(X, y)
    1.0
    >>> model.predict(np.array([1,2,3,4,5]))
    array([-1.15907818, -1.15907818, -1.15907818, -1.15907818, -1.15907818])
    ```
    """

    equations_: pd.DataFrame | list[pd.DataFrame] | None
    n_features_in_: int
    feature_names_in_: ArrayLike[str]
    display_feature_names_in_: ArrayLike[str]
    complexity_of_variables_: int | float | list[int | float] | None
    X_units_: ArrayLike[str] | None
    y_units_: str | ArrayLike[str] | None
    nout_: int
    selection_mask_: NDArray[np.bool_] | None
    run_id_: str
    output_directory_: str
    julia_state_stream_: NDArray[np.uint8] | None
    julia_options_stream_: NDArray[np.uint8] | None
    logger_: AnyValue | None
    equation_file_contents_: list[pd.DataFrame] | None
    show_pickle_warnings_: bool

    def __init__(
        self,
        model_selection: Literal["best", "accuracy", "score"] = "best",
        *,
        binary_operators: list[str] | None = None,
        unary_operators: list[str] | None = None,
        expression_spec: AbstractExpressionSpec | None = None,
        niterations: int = 100,
        populations: int = 31,
        population_size: int = 27,
        max_evals: int | None = None,
        maxsize: int = 30,
        maxdepth: int | None = None,
        warmup_maxsize_by: float | None = None,
        timeout_in_seconds: float | None = None,
        constraints: dict[str, int | tuple[int, int]] | None = None,
        nested_constraints: dict[str, dict[str, int]] | None = None,
        elementwise_loss: str | None = None,
        loss_function: str | None = None,
        loss_function_expression: str | None = None,
        loss_scale: Literal["log", "linear"] = "log",
        complexity_of_operators: dict[str, int | float] | None = None,
        complexity_of_constants: int | float | None = None,
        complexity_of_variables: int | float | list[int | float] | None = None,
        complexity_mapping: str | None = None,
        parsimony: float = 0.0,
        dimensional_constraint_penalty: float | None = None,
        dimensionless_constants_only: bool = False,
        use_frequency: bool = True,
        use_frequency_in_tournament: bool = True,
        adaptive_parsimony_scaling: float = 1040.0,
        alpha: float = 3.17,
        annealing: bool = False,
        early_stop_condition: float | str | None = None,
        ncycles_per_iteration: int = 380,
        fraction_replaced: float = 0.00036,
        fraction_replaced_hof: float = 0.0614,
        weight_add_node: float = 2.47,
        weight_insert_node: float = 0.0112,
        weight_delete_node: float = 0.870,
        weight_do_nothing: float = 0.273,
        weight_mutate_constant: float = 0.0346,
        weight_mutate_operator: float = 0.293,
        weight_swap_operands: float = 0.198,
        weight_rotate_tree: float = 4.26,
        weight_randomize: float = 0.000502,
        weight_simplify: float = 0.00209,
        weight_optimize: float = 0.0,
        crossover_probability: float = 0.0259,
        skip_mutation_failures: bool = True,
        migration: bool = True,
        hof_migration: bool = True,
        topn: int = 12,
        should_simplify: bool = True,
        should_optimize_constants: bool = True,
        optimizer_algorithm: Literal["BFGS", "NelderMead"] = "BFGS",
        optimizer_nrestarts: int = 2,
        optimizer_f_calls_limit: int | None = None,
        optimize_probability: float = 0.14,
        optimizer_iterations: int = 8,
        perturbation_factor: float = 0.129,
        probability_negate_constant: float = 0.00743,
        tournament_selection_n: int = 15,
        tournament_selection_p: float = 0.982,
        parallelism: (
            Literal["serial", "multithreading", "multiprocessing"] | None
        ) = None,
        procs: int | None = None,
        cluster_manager: (
            Literal["slurm", "pbs", "lsf", "sge", "qrsh", "scyld", "htc"] | None
        ) = None,
        heap_size_hint_in_bytes: int | None = None,
        batching: bool = False,
        batch_size: int = 50,
        fast_cycle: bool = False,
        turbo: bool = False,
        bumper: bool = False,
        precision: Literal[16, 32, 64] = 32,
        autodiff_backend: Literal["Zygote"] | None = None,
        random_state: int | np.random.RandomState | None = None,
        deterministic: bool = False,
        warm_start: bool = False,
        verbosity: int = 1,
        update_verbosity: int | None = None,
        print_precision: int = 5,
        progress: bool = True,
        logger_spec: AbstractLoggerSpec | None = None,
        input_stream: str = "stdin",
        run_id: str | None = None,
        output_directory: str | None = None,
        temp_equation_file: bool = False,
        tempdir: str | None = None,
        delete_tempfiles: bool = True,
        update: bool = False,
        output_jax_format: bool = False,
        output_torch_format: bool = False,
        extra_sympy_mappings: dict[str, Callable] | None = None,
        extra_torch_mappings: dict[Callable, Callable] | None = None,
        extra_jax_mappings: dict[Callable, str] | None = None,
        denoise: bool = False,
        select_k_features: int | None = None,
        **kwargs,
    ):
        # Hyperparameters
        # - Model search parameters
        self.model_selection = model_selection
        self.binary_operators = binary_operators
        self.unary_operators = unary_operators
        self.expression_spec = expression_spec
        self.niterations = niterations
        self.populations = populations
        self.population_size = population_size
        self.ncycles_per_iteration = ncycles_per_iteration
        # - Equation Constraints
        self.maxsize = maxsize
        self.maxdepth = maxdepth
        self.constraints = constraints
        self.nested_constraints = nested_constraints
        self.warmup_maxsize_by = warmup_maxsize_by
        self.should_simplify = should_simplify
        # - Early exit conditions:
        self.max_evals = max_evals
        self.timeout_in_seconds = timeout_in_seconds
        self.early_stop_condition = early_stop_condition
        # - Loss parameters
        self.elementwise_loss = elementwise_loss
        self.loss_function = loss_function
        self.loss_function_expression = loss_function_expression
        self.loss_scale = loss_scale
        self.complexity_of_operators = complexity_of_operators
        self.complexity_of_constants = complexity_of_constants
        self.complexity_of_variables = complexity_of_variables
        self.complexity_mapping = complexity_mapping
        self.parsimony = parsimony
        self.dimensional_constraint_penalty = dimensional_constraint_penalty
        self.dimensionless_constants_only = dimensionless_constants_only
        self.use_frequency = use_frequency
        self.use_frequency_in_tournament = use_frequency_in_tournament
        self.adaptive_parsimony_scaling = adaptive_parsimony_scaling
        self.alpha = alpha
        self.annealing = annealing
        # - Evolutionary search parameters
        # -- Mutation parameters
        self.weight_add_node = weight_add_node
        self.weight_insert_node = weight_insert_node
        self.weight_delete_node = weight_delete_node
        self.weight_do_nothing = weight_do_nothing
        self.weight_mutate_constant = weight_mutate_constant
        self.weight_mutate_operator = weight_mutate_operator
        self.weight_swap_operands = weight_swap_operands
        self.weight_rotate_tree = weight_rotate_tree
        self.weight_randomize = weight_randomize
        self.weight_simplify = weight_simplify
        self.weight_optimize = weight_optimize
        self.crossover_probability = crossover_probability
        self.skip_mutation_failures = skip_mutation_failures
        # -- Migration parameters
        self.migration = migration
        self.hof_migration = hof_migration
        self.fraction_replaced = fraction_replaced
        self.fraction_replaced_hof = fraction_replaced_hof
        self.topn = topn
        # -- Constants parameters
        self.should_optimize_constants = should_optimize_constants
        self.optimizer_algorithm = optimizer_algorithm
        self.optimizer_nrestarts = optimizer_nrestarts
        self.optimizer_f_calls_limit = optimizer_f_calls_limit
        self.optimize_probability = optimize_probability
        self.optimizer_iterations = optimizer_iterations
        self.perturbation_factor = perturbation_factor
        self.probability_negate_constant = probability_negate_constant
        # -- Selection parameters
        self.tournament_selection_n = tournament_selection_n
        self.tournament_selection_p = tournament_selection_p
        # -- Performance parameters
        self.parallelism = parallelism
        self.procs = procs
        self.cluster_manager = cluster_manager
        self.heap_size_hint_in_bytes = heap_size_hint_in_bytes
        self.batching = batching
        self.batch_size = batch_size
        self.fast_cycle = fast_cycle
        self.turbo = turbo
        self.bumper = bumper
        self.precision = precision
        self.autodiff_backend = autodiff_backend
        self.random_state = random_state
        self.deterministic = deterministic
        self.warm_start = warm_start
        # Additional runtime parameters
        # - Runtime user interface
        self.verbosity = verbosity
        self.update_verbosity = update_verbosity
        self.print_precision = print_precision
        self.progress = progress
        self.logger_spec = logger_spec
        self.input_stream = input_stream
        # - Project management
        self.run_id = run_id
        self.output_directory = output_directory
        self.temp_equation_file = temp_equation_file
        self.tempdir = tempdir
        self.delete_tempfiles = delete_tempfiles
        self.update = update
        self.output_jax_format = output_jax_format
        self.output_torch_format = output_torch_format
        self.extra_sympy_mappings = extra_sympy_mappings
        self.extra_jax_mappings = extra_jax_mappings
        self.extra_torch_mappings = extra_torch_mappings
        # Pre-modelling transformation
        self.denoise = denoise
        self.select_k_features = select_k_features

        # Once all valid parameters have been assigned handle the
        # deprecated kwargs
        if len(kwargs) > 0:  # pragma: no cover
            for k, v in kwargs.items():
                # Handle renamed kwargs
                if k in DEPRECATED_KWARGS:
                    updated_kwarg_name = DEPRECATED_KWARGS[k]
                    setattr(self, updated_kwarg_name, v)
                    warnings.warn(
                        f"`{k}` has been renamed to `{updated_kwarg_name}` in PySRRegressor. "
                        "Please use that instead.",
                        FutureWarning,
                    )
                elif k == "multithreading":
                    # Specific advice given in `_map_parallelism_params`
                    self.multithreading: bool | None = v
                # Handle kwargs that have been moved to the fit method
                elif k in ["weights", "variable_names", "Xresampled"]:
                    warnings.warn(
                        f"`{k}` is a data-dependent parameter and should be passed when fit is called. "
                        f"Ignoring parameter; please pass `{k}` during the call to fit instead.",
                        FutureWarning,
                    )
                elif k == "julia_project":
                    warnings.warn(
                        "The `julia_project` parameter has been deprecated. To use a custom "
                        "julia project, please see `https://ai.damtp.cam.ac.uk/pysr/backend`.",
                        FutureWarning,
                    )
                elif k == "julia_kwargs":
                    warnings.warn(
                        "The `julia_kwargs` parameter has been deprecated. To pass custom "
                        "keyword arguments to the julia backend, you should use environment variables. "
                        "See the Julia documentation for more information.",
                        FutureWarning,
                    )
                else:
                    suggested_keywords = _suggest_keywords(PySRRegressor, k)
                    err_msg = (
                        f"`{k}` is not a valid keyword argument for PySRRegressor."
                    )
                    if len(suggested_keywords) > 0:
                        err_msg += f" Did you mean {', '.join(map(lambda s: f'`{s}`', suggested_keywords))}?"
                    raise TypeError(err_msg)

    @classmethod
    def from_file(
        cls,
        equation_file: None = None,  # Deprecated
        *,
        run_directory: PathLike,
        binary_operators: list[str] | None = None,
        unary_operators: list[str] | None = None,
        n_features_in: int | None = None,
        feature_names_in: ArrayLike[str] | None = None,
        selection_mask: NDArray[np.bool_] | None = None,
        nout: int = 1,
        **pysr_kwargs,
    ) -> "PySRRegressor":
        """
        Create a model from a saved model checkpoint or equation file.

        Parameters
        ----------
        run_directory : str
            The directory containing outputs from a previous run.
            This is of the form `[output_directory]/[run_id]`.
            Default is `None`.
        binary_operators : list[str]
            The same binary operators used when creating the model.
            Not needed if loading from a pickle file.
        unary_operators : list[str]
            The same unary operators used when creating the model.
            Not needed if loading from a pickle file.
        n_features_in : int
            Number of features passed to the model.
            Not needed if loading from a pickle file.
        feature_names_in : list[str]
            Names of the features passed to the model.
            Not needed if loading from a pickle file.
        selection_mask : NDArray[np.bool_]
            If using `select_k_features`, you must pass `model.selection_mask_` here.
            Not needed if loading from a pickle file.
        nout : int
            Number of outputs of the model.
            Not needed if loading from a pickle file.
            Default is `1`.
        **pysr_kwargs : dict
            Any other keyword arguments to initialize the PySRRegressor object.
            These will overwrite those stored in the pickle file.
            Not needed if loading from a pickle file.

        Returns
        -------
        model : PySRRegressor
            The model with fitted equations.
        """
        if equation_file is not None:
            raise ValueError(
                "Passing `equation_file` is deprecated and no longer compatible with "
                "the most recent versions of PySR's backend. Please pass `run_directory` "
                "instead, which contains all checkpoint files."
            )

        pkl_filename = Path(run_directory) / "checkpoint.pkl"
        if pkl_filename.exists():
            pysr_logger.info(f"Attempting to load model from {pkl_filename}...")
            assert binary_operators is None
            assert unary_operators is None
            assert n_features_in is None
            with open(pkl_filename, "rb") as f:
                model = cast("PySRRegressor", pkl.load(f))

            # Update any parameters if necessary, such as
            # extra_sympy_mappings:
            model.set_params(**pysr_kwargs)

            if "equations_" not in model.__dict__ or model.equations_ is None:
                model.refresh()

            if model.expression_spec is not None:
                warnings.warn(
                    "Loading model from checkpoint file with a non-default expression spec "
                    "is not fully supported as it relies on dynamic objects. This may result in unexpected behavior.",
                )

            return model
        else:
            pysr_logger.info(
                f"Checkpoint file {pkl_filename} does not exist. "
                "Attempting to recreate model from scratch..."
            )
            csv_filename = Path(run_directory) / "hall_of_fame.csv"
            csv_filename_bak = Path(run_directory) / "hall_of_fame.csv.bak"
            if not csv_filename.exists() and not csv_filename_bak.exists():
                raise FileNotFoundError(
                    f"Hall of fame file `{csv_filename}` or `{csv_filename_bak}` does not exist. "
                    "Please pass a `run_directory` containing a valid checkpoint file."
                )
            assert binary_operators is not None or unary_operators is not None
            assert n_features_in is not None
            model = cls(
                binary_operators=binary_operators,
                unary_operators=unary_operators,
                **pysr_kwargs,
            )
            model.nout_ = nout
            model.n_features_in_ = n_features_in

            if feature_names_in is None:
                model.feature_names_in_ = np.array(
                    [f"x{i}" for i in range(n_features_in)]
                )
                model.display_feature_names_in_ = np.array(
                    [f"x{_subscriptify(i)}" for i in range(n_features_in)]
                )
            else:
                assert len(feature_names_in) == n_features_in
                model.feature_names_in_ = feature_names_in
                model.display_feature_names_in_ = feature_names_in

            if selection_mask is None:
                model.selection_mask_ = np.ones(n_features_in, dtype=np.bool_)
            else:
                model.selection_mask_ = selection_mask

            model.refresh(run_directory=run_directory)

            return model

    def __repr__(self) -> str:
        """
        Print all current equations fitted by the model.

        The string `>>>>` denotes which equation is selected by the
        `model_selection`.
        """
        if not hasattr(self, "equations_") or self.equations_ is None:
            return "PySRRegressor.equations_ = None"

        output = "PySRRegressor.equations_ = [\n"

        equations = self.equations_
        if not isinstance(equations, list):
            all_equations = [equations]
        else:
            all_equations = equations

        for i, equations in enumerate(all_equations):
            selected = pd.Series([""] * len(equations), index=equations.index)
            chosen_row = idx_model_selection(equations, self.model_selection)
            selected[chosen_row] = ">>>>"
            repr_equations = pd.DataFrame(
                dict(
                    pick=selected,
                    **(
                        {"score": equations["score"]}
                        if "score" in equations.columns
                        else {}
                    ),
                    equation=equations["equation"],
                    loss=equations["loss"],
                    complexity=equations["complexity"],
                )
            )

            if len(all_equations) > 1:
                output += "[\n"

            for line in repr_equations.__repr__().split("\n"):
                output += "\t" + line + "\n"

            if len(all_equations) > 1:
                output += "]"

            if i < len(all_equations) - 1:
                output += ", "

        output += "]"
        return output

    def __getstate__(self) -> dict[str, Any]:
        """
        Handle pickle serialization for PySRRegressor.

        The Scikit-learn standard requires estimators to be serializable via
        `pickle.dumps()`. However, some attributes do not support pickling
        and need to be hidden, such as the JAX and Torch representations.
        """
        state = self.__dict__
        show_pickle_warning = not (
            "show_pickle_warnings_" in state and not state["show_pickle_warnings_"]
        )
        state_keys_containing_lambdas = ["extra_sympy_mappings", "extra_torch_mappings"]
        for state_key in state_keys_containing_lambdas:
            warn_msg = (
                f"`{state_key}` cannot be pickled and will be removed from the "
                "serialized instance. When loading the model, please redefine "
                f"`{state_key}` at runtime."
            )
            if state[state_key] is not None:
                if show_pickle_warning:
                    warnings.warn(warn_msg)
                else:
                    pysr_logger.debug(warn_msg)
        state_keys_to_clear = state_keys_containing_lambdas
        state_keys_to_clear.append("logger_")
        pickled_state = {
            key: (None if key in state_keys_to_clear else value)
            for key, value in state.items()
        }
        if ("equations_" in pickled_state) and (
            pickled_state["equations_"] is not None
        ):
            pickled_state["output_torch_format"] = False
            pickled_state["output_jax_format"] = False
            if self.nout_ == 1:
                pickled_columns = ~pickled_state["equations_"].columns.isin(
                    ["jax_format", "torch_format"]
                )
                pickled_state["equations_"] = (
                    pickled_state["equations_"].loc[:, pickled_columns].copy()
                )
            else:
                pickled_columns = [
                    ~dataframe.columns.isin(["jax_format", "torch_format"])
                    for dataframe in pickled_state["equations_"]
                ]
                pickled_state["equations_"] = [
                    dataframe.loc[:, signle_pickled_columns]
                    for dataframe, signle_pickled_columns in zip(
                        pickled_state["equations_"], pickled_columns
                    )
                ]
        return pickled_state

    def _checkpoint(self):
        """Save the model's current state to a checkpoint file.

        This should only be used internally by PySRRegressor.
        """
        # Save model state:
        self.show_pickle_warnings_ = False
        with open(self.get_pkl_filename(), "wb") as f:
            try:
                pkl.dump(self, f)
            except Exception as e:
                pysr_logger.debug(f"Error checkpointing model: {e}")
        self.show_pickle_warnings_ = True

    def get_pkl_filename(self) -> Path:
        path = Path(self.output_directory_) / self.run_id_ / "checkpoint.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def equations(self):  # pragma: no cover
        warnings.warn(
            "PySRRegressor.equations is now deprecated. "
            "Please use PySRRegressor.equations_ instead.",
            FutureWarning,
        )
        return self.equations_

    @property
    def julia_options_(self):
        """The deserialized julia options."""
        return jl_deserialize(self.julia_options_stream_)

    @property
    def julia_state_(self):
        """The deserialized state."""
        return cast(
            Union[Tuple[VectorValue, AnyValue], None],
            jl_deserialize(self.julia_state_stream_),
        )

    @property
    def raw_julia_state_(self):
        warnings.warn(
            "PySRRegressor.raw_julia_state_ is now deprecated. "
            "Please use PySRRegressor.julia_state_ instead, or julia_state_stream_ "
            "for the raw stream of bytes.",
            FutureWarning,
        )
        return self.julia_state_

    @property
    def expression_spec_(self):
        return self.expression_spec or ExpressionSpec()

    def get_best(
        self, index: int | list[int] | None = None
    ) -> pd.Series | list[pd.Series]:
        """
        Get best equation using `model_selection`.

        Parameters
        ----------
        index : int | list[int]
            If you wish to select a particular equation from `self.equations_`,
            give the row number here. This overrides the `model_selection`
            parameter. If there are multiple output features, then pass
            a list of indices with the order the same as the output feature.

        Returns
        -------
        best_equation : pandas.Series
            Dictionary representing the best expression found.

        Raises
        ------
        NotImplementedError
            Raised when an invalid model selection strategy is provided.
        """
        check_is_fitted(self, attributes=["equations_"])

        if index is not None:
            if isinstance(self.equations_, list):
                assert isinstance(
                    index, list
                ), "With multiple output features, index must be a list."
                return [eq.iloc[i] for eq, i in zip(self.equations_, index)]
            else:
                equations_ = cast(pd.DataFrame, self.equations_)
                return cast(pd.Series, equations_.iloc[index])

        if isinstance(self.equations_, list):
            return [
                cast(pd.Series, eq.loc[idx_model_selection(eq, self.model_selection)])
                for eq in self.equations_
            ]
        else:
            equations_ = cast(pd.DataFrame, self.equations_)
            return cast(
                pd.Series,
                equations_.loc[idx_model_selection(equations_, self.model_selection)],
            )

    @property
    def equation_file_(self):
        raise NotImplementedError(
            "PySRRegressor.equation_file_ is now deprecated. "
            "Please use PySRRegressor.output_directory_ and PySRRegressor.run_id_ "
            "instead. For loading, you should pass `run_directory`."
        )

    def _setup_equation_file(self):
        """Set the pathname of the output directory."""
        if self.warm_start and (
            hasattr(self, "run_id_") or hasattr(self, "output_directory_")
        ):
            assert hasattr(self, "output_directory_")
            assert hasattr(self, "run_id_")
            if self.run_id is not None:
                assert self.run_id_ == self.run_id
            if self.output_directory is not None:
                assert self.output_directory_ == self.output_directory
        else:
            self.output_directory_ = (
                tempfile.mkdtemp()
                if self.temp_equation_file
                else (
                    "outputs"
                    if self.output_directory is None
                    else self.output_directory
                )
            )
            self.run_id_ = (
                cast(str, SymbolicRegression.SearchUtilsModule.generate_run_id())
                if self.run_id is None
                else self.run_id
            )
            if self.temp_equation_file:
                assert self.output_directory is None

    def _clear_equation_file_contents(self):
        self.equation_file_contents_ = None

    def _validate_and_modify_params(self) -> _DynamicallySetParams:
        """
        Ensure parameters passed at initialization are valid.

        Also returns a dictionary of parameters to update from their
        values given at initialization.

        Returns
        -------
        packed_modified_params : dict
            Dictionary of parameters to modify from their initialized
            values. For example, default parameters are set here
            when a parameter is left set to `None`.
        """
        # Immutable parameter validation
        # Ensure instance parameters are allowable values:
        if self.tournament_selection_n > self.population_size:
            raise ValueError(
                "`tournament_selection_n` parameter must be smaller than `population_size`."
            )

        if self.maxsize > 40:
            warnings.warn(
                "Note: Using a large maxsize for the equation search will be "
                "exponentially slower and use significant memory."
            )
        elif self.maxsize < 7:
            raise ValueError("PySR requires a maxsize of at least 7")

        # NotImplementedError - Values that could be supported at a later time
        if self.optimizer_algorithm not in VALID_OPTIMIZER_ALGORITHMS:
            raise NotImplementedError(
                f"PySR currently only supports the following optimizer algorithms: {VALID_OPTIMIZER_ALGORITHMS}"
            )

        param_container = _DynamicallySetParams(
            binary_operators=["+", "*", "-", "/"],
            unary_operators=[],
            maxdepth=self.maxsize,
            constraints={},
            batch_size=1,
            update_verbosity=int(self.verbosity),
            progress=self.progress,
            warmup_maxsize_by=0.0,
        )

        for param_name in map(lambda x: x.name, fields(_DynamicallySetParams)):
            user_param_value = getattr(self, param_name)
            if user_param_value is None:
                # Leave as the default in DynamicallySetParams
                ...
            else:
                # If user has specified it, we will override the default.
                # However, there are some special cases to mutate it:
                new_param_value = _mutate_parameter(param_name, user_param_value)
                setattr(param_container, param_name, new_param_value)
        # TODO: This should just be part of the __init__ of _DynamicallySetParams

        assert (
            len(param_container.binary_operators) > 0
            or len(param_container.unary_operators) > 0
        ), "At least one operator must be provided."

        return param_container

    def _validate_and_set_fit_params(
        self,
        X,
        y,
        Xresampled,
        weights,
        variable_names,
        complexity_of_variables,
        X_units,
        y_units,
    ) -> tuple[
        ndarray,
        ndarray,
        ndarray | None,
        ndarray | None,
        ArrayLike[str],
        int | float | list[int | float] | None,
        ArrayLike[str] | None,
        str | ArrayLike[str] | None,
    ]:
        """
        验证传递给:term:`fit`方法的参数。

        此方法还设置`nout_`属性。

        Parameters
        ----------
        X : ndarray | pandas.DataFrame
            形状为`(n_samples, n_features)`的训练数据。
        y : ndarray | pandas.DataFrame}
            形状为`(n_samples,)`或`(n_samples, n_targets)`的目标值。
            如有必要，将被转换为`X`的数据类型。
        Xresampled : ndarray | pandas.DataFrame
            用于去噪的重采样训练数据，
            形状为`(n_resampled, n_features)`。
        weights : ndarray | pandas.DataFrame
            与`y`形状相同的权重数组。
            每个元素表示如何对均方误差损失进行加权
            对于y的特定元素。
        variable_names : ndarray of length n_features
            训练数据集中每个特征的名称，`X`。
        complexity_of_variables : int | float | list[int | float]
            训练数据集中每个特征的复杂度，`X`。
        X_units : list[str] of length n_features
            训练数据集中每个特征的单位，`X`。
        y_units : str | list[str] of length n_out
            训练数据集中每个特征的单位，`y`。

        Returns
        -------
        X_validated : ndarray of shape (n_samples, n_features)
            验证后的训练数据。
        y_validated : ndarray of shape (n_samples,) or (n_samples, n_targets)
            验证后的目标数据。
        Xresampled : ndarray of shape (n_resampled, n_features)
            验证后的用于去噪的重采样训练数据。
        variable_names_validated : list[str] of length n_features
            验证后的`X`中每个特征的变量名列表。
        X_units : list[str] of length n_features
            验证后的`X`的单位。
        y_units : str | list[str] of length n_out
            验证后的`y`的单位。

        """
        # 检查X是否为pandas DataFrame
        if isinstance(X, pd.DataFrame):
            # 如果X是DataFrame且variable_names已提供，则重置variable_names并发出警告
            if variable_names:
                variable_names = None
                warnings.warn(
                    "`variable_names`已重置为`None`，因为`X`是DataFrame。"
                    "将使用DataFrame的列名。"
                )

            # 检查DataFrame的列名是否包含空格，如果有则替换为下划线并发出警告
            if (
                pd.api.types.is_object_dtype(X.columns)
                and X.columns.str.contains(" ").any()
            ):
                X.columns = X.columns.str.replace(" ", "_")
                warnings.warn(
                    "DataFrame列名中的空格不受支持。"
                    "空格已被替换为下划线。\n"
                    "请将列重命名为有效名称。"
                )
        # 如果X不是DataFrame但variable_names包含空格，则替换空格为下划线并发出警告
        elif variable_names and any([" " in name for name in variable_names]):
            variable_names = [name.replace(" ", "_") for name in variable_names]
            warnings.warn(
                "`variable_names`中的空格不受支持。"
                "空格已被替换为下划线。\n"
                "请使用有效名称。"
            )

        # 检查complexity_of_variables是否在fit方法和__init__方法中都被设置，如果是则抛出错误
        if (
            complexity_of_variables is not None
            and self.complexity_of_variables is not None
        ):
            raise ValueError(
                "不能在`fit`和`__init__`中都设置`complexity_of_variables`。"
                "在`__init__`中传递以设置全局默认值，或在`fit`中传递以单独设置每个变量。"
            )
        # 如果在fit方法中设置了complexity_of_variables，则使用该值
        elif complexity_of_variables is not None:
            complexity_of_variables = complexity_of_variables
        # 如果在__init__方法中设置了complexity_of_variables，则使用该值
        elif self.complexity_of_variables is not None:
            complexity_of_variables = self.complexity_of_variables
        # 如果都没有设置，则设为None
        else:
            complexity_of_variables = None

        # 通过sklearn进行数据验证和特征名称获取
        # 此方法设置n_features_in_属性
        # 如果Xresampled不为None，则验证Xresampled数组
        if Xresampled is not None:
            Xresampled = check_array(Xresampled)
        # 如果weights不为None，则验证weights数组并检查与y的一致性
        if weights is not None:
            weights = check_array(weights, ensure_2d=False)
            check_consistent_length(weights, y)
        # 验证X和y的数据
        X, y = self._validate_data_X_y(X, y)
        # 检查并获取特征名称
        self.feature_names_in_ = _safe_check_feature_names_in(
            self, variable_names, generate_names=False
        )

        # 如果没有特征名称，则生成默认名称
        if self.feature_names_in_ is None:
            # 生成默认特征名称"x0", "x1", ...
            self.feature_names_in_ = np.array([f"x{i}" for i in range(X.shape[1])])
            # 生成显示用的下标特征名称"x₀", "x₁", ...
            self.display_feature_names_in_ = np.array(
                [f"x{_subscriptify(i)}" for i in range(X.shape[1])]
            )
            variable_names = self.feature_names_in_
        else:
            # 如果已有特征名称，则设置显示用的特征名称
            self.display_feature_names_in_ = self.feature_names_in_
            variable_names = self.feature_names_in_

        # 处理多输出数据
        # 如果y是一维数组或二维但只有一列，则将其reshape为一维
        if len(y.shape) == 1 or (len(y.shape) == 2 and y.shape[1] == 1):
            y = y.reshape(-1)
        # 如果y是二维数组且有多列，则设置输出数量nout_
        elif len(y.shape) == 2:
            self.nout_ = y.shape[1]
        # 如果y的形状不支持，则抛出NotImplementedError
        else:
            raise NotImplementedError("不支持的y形状!")

        # 深拷贝复杂度、X单位和y单位变量
        self.complexity_of_variables_ = copy.deepcopy(complexity_of_variables)
        self.X_units_ = copy.deepcopy(X_units)
        self.y_units_ = copy.deepcopy(y_units)

        # 返回验证后的参数
        return (
            X,
            y,
            Xresampled,
            weights,
            variable_names,
            complexity_of_variables,
            X_units,
            y_units,
        )

    def _validate_data_X_y(self, X: Any, y: Any) -> tuple[ndarray, ndarray]:
        if OLD_SKLEARN:
            raw_out = self._validate_data(X=X, y=y, reset=True, multi_output=True)  # type: ignore
        else:
            raw_out = validate_data(self, X=X, y=y, reset=True, multi_output=True)  # type: ignore
        return cast(Tuple[ndarray, ndarray], raw_out)

    def _validate_data_X(self, X: Any) -> ndarray:
        if OLD_SKLEARN:
            raw_out = self._validate_data(X=X, reset=False)  # type: ignore
        else:
            raw_out = validate_data(self, X=X, reset=False)  # type: ignore
        return cast(ndarray, raw_out)

    def _get_precision_mapped_dtype(self, X: np.ndarray) -> type:
        is_complex = np.issubdtype(X.dtype, np.complexfloating)
        is_real = not is_complex
        if is_real:
            return {16: np.float16, 32: np.float32, 64: np.float64}[self.precision]
        else:
            return {32: np.complex64, 64: np.complex128}[self.precision]

    def _pre_transform_training_data(
        self,
        X: ndarray,
        y: ndarray,
        Xresampled: ndarray | None,
        variable_names: ArrayLike[str],
        complexity_of_variables: int | float | list[int | float] | None,
        X_units: ArrayLike[str] | None,
        y_units: ArrayLike[str] | str | None,
        random_state: np.random.RandomState,
    ):
        """
        在拟合符号回归器之前转换训练数据。

        此方法还更新/设置`selection_mask_`属性。

        Parameters
        ----------
        X : ndarray
            形状为(n_samples, n_features)的训练数据。
        y : ndarray
            形状为(n_samples,)或(n_samples, n_targets)的目标值。
            如有必要，将被转换为X的数据类型。
        Xresampled : ndarray | None
            重采样训练数据，形状为`(n_resampled, n_features)`，
            用于去噪。
        variable_names : list[str]
            训练数据集`X`中每个变量的名称。
            长度为`n_features`。
        complexity_of_variables : int | float | list[int | float] | None
            训练数据集`X`中每个变量的复杂度。
        X_units : list[str]
            训练数据集`X`中每个变量的单位。
        y_units : str | list[str]
            训练数据集`y`中每个变量的单位。
        random_state : int | np.RandomState
            传递一个整数以在多次函数调用中获得可重现的结果。
            请参见:term:`Glossary <random_state>`。默认为`None`。

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            转换后的训练数据。如果`self.denoise`为`True`，
            且`Xresampled is not None`，则n_samples等于`Xresampled.shape[0]`，
            否则等于`X.shape[0]`。如果`self.select_k_features is not None`，
            则n_features等于`self.select_k_features`，
            否则等于`X.shape[1]`
        y_transformed : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            转换后的目标数据。如果`self.denoise`为`True`，
            且`Xresampled is not None`，则n_samples等于`Xresampled.shape[0]`，
            否则等于`X.shape[0]`。
        variable_names_transformed : list[str] of length n_features
            转换后数据集`X_transformed`中每个变量的名称。
        X_units_transformed : list[str] of length n_features
            转换后数据集中每个变量的单位。
        y_units_transformed : str | list[str] of length n_out
            转换后数据集中每个变量的单位。
        """
        # 特征选择转换
        # 如果启用了特征选择（select_k_features不为None）
        if self.select_k_features:
            # 运行特征选择算法，选择k个最佳特征
            selection_mask = run_feature_selection(
                X, y, self.select_k_features, random_state=random_state
            )
            # 根据选择掩码过滤X数据，只保留选中的特征
            X = X[:, selection_mask]

            # 如果提供了Xresampled数据，也对它进行相同的特征选择
            if Xresampled is not None:
                Xresampled = Xresampled[:, selection_mask]

            # 根据选择掩码减少variable_names，只保留选中的特征名称
            variable_names = cast(
                ArrayLike[str],
                [
                    variable_names[i]
                    for i in range(len(variable_names))
                    if selection_mask[i]
                ],
            )

            # 如果complexity_of_variables是列表，也对其进行特征选择过滤
            if isinstance(complexity_of_variables, list):
                complexity_of_variables = [
                    complexity_of_variables[i]
                    for i in range(len(complexity_of_variables))
                    if selection_mask[i]
                ]
                # 更新模型的复杂度变量属性
                self.complexity_of_variables_ = copy.deepcopy(complexity_of_variables)

            # 如果提供了X_units，也对其进行特征选择过滤
            if X_units is not None:
                X_units = cast(
                    ArrayLike[str],
                    [X_units[i] for i in range(len(X_units)) if selection_mask[i]],
                )
                # 更新模型的X_units属性
                self.X_units_ = copy.deepcopy(X_units)

            # 重新执行数据验证和特征名称更新
            X, y = self._validate_data_X_y(X, y)
            # 使用选定的变量名称更新特征名称
            self.selection_mask_ = selection_mask
            self.feature_names_in_ = _check_feature_names_in(self, variable_names)
            self.display_feature_names_in_ = self.feature_names_in_
            # 记录使用的特征信息
            pysr_logger.info(f"Using features {self.feature_names_in_}")

        # 去噪转换
        # 如果启用了去噪功能
        if self.denoise:
            # 如果输出变量数量大于1（多输出情况）
            if self.nout_ > 1:
                # 使用多输出去噪函数处理数据
                X, y = multi_denoise(
                    X, y, Xresampled=Xresampled, random_state=random_state
                )
            else:
                # 使用单输出去噪函数处理数据
                X, y = denoise(X, y, Xresampled=Xresampled, random_state=random_state)

        # 返回转换后的数据和相关参数
        return X, y, variable_names, complexity_of_variables, X_units, y_units

    def _run(
        self,
        X: ndarray,
        y: ndarray,
        runtime_params: _DynamicallySetParams,
        weights: ndarray | None,
        category: ndarray | None,
        seed: int,
    ):
        """
        在Julia后端运行符号回归拟合过程。

        Parameters
        ----------
        X : ndarray
            形状为`(n_samples, n_features)`的训练数据。
        y : ndarray
            形状为`(n_samples,)`或`(n_samples, n_targets)`的目标值。
            如有必要，将被转换为`X`的数据类型。
        runtime_params : DynamicallySetParams
            从__init__中传递的一些参数的动态设置版本。
        weights : ndarray | None
            与`y`形状相同的权重数组。
            每个元素表示如何对均方误差损失进行加权
            对于y的特定元素。
        category : ndarray | None
            如果`expression_spec`是`ParametricExpressionSpec`，则此参数
            应该是表示`X`中每个样本类别的整数列表。
        seed : int
            Julia后端进程的随机种子。

        Returns
        -------
        self : object
            带有拟合属性的`self`引用。

        Raises
        ------
        ImportError
            当Julia后端无法导入包时引发。
        """
        # 需要全局化，因为我们不想为PySRRegressor的每个新实例重新创建/重新启动Julia
        global ALREADY_RAN

        # 这些参数可能与初始化时指定的参数不同，所以我们在这里本地定义它们：
        binary_operators = runtime_params.binary_operators
        unary_operators = runtime_params.unary_operators
        constraints = runtime_params.constraints

        nested_constraints = self.nested_constraints
        complexity_of_operators = self.complexity_of_operators
        complexity_of_variables = self.complexity_of_variables_
        cluster_manager = self.cluster_manager

        # 启动Julia后端进程
        # 如果尚未运行且更新详细程度不为0，则显示编译信息
        if not ALREADY_RAN and runtime_params.update_verbosity != 0:
            pysr_logger.info("正在编译Julia后端...")

        # 映射并行参数，获取并行模式和进程数
        parallelism, numprocs = _map_parallelism_params(
            self.parallelism, self.procs, getattr(self, "multithreading", None)
        )

        # 如果要求确定性搜索但并行模式不是串行，则抛出错误
        if self.deterministic and parallelism != "serial":
            raise ValueError(
                "为确保确定性搜索，必须设置`parallelism='serial'`。"
                "此外，请确保将`random_state`设置为种子。"
            )
        # 如果设置了随机状态但并行模式不是串行或未设置确定性搜索，则发出警告
        if self.random_state is not None and (
            parallelism != "serial" or not self.deterministic
        ):
            warnings.warn(
                "注意：设置`random_state`而不同时设置`deterministic=True`"
                "和`parallelism='serial'`将导致非确定性搜索。"
            )

        # 如果指定了集群管理器
        if cluster_manager is not None:
            # 检查并行模式是否为多进程
            if parallelism != "multiprocessing":
                raise ValueError(
                    "要使用集群管理器，必须设置`parallelism='multiprocessing'`。"
                )
            # 加载集群管理器
            cluster_manager = _load_cluster_manager(cluster_manager)

        # TODO(mcranmer): 这些函数应该是这个类的一部分。
        # 可能创建内联操作符
        binary_operators, unary_operators = _maybe_create_inline_operators(
            binary_operators=binary_operators,
            unary_operators=unary_operators,
            extra_sympy_mappings=self.extra_sympy_mappings,
            expression_spec=self.expression_spec_,
        )
        # 如果有约束条件，则处理约束
        if constraints is not None:
            _constraints = _process_constraints(
                binary_operators=binary_operators,
                unary_operators=unary_operators,
                constraints=constraints,
            )
            # 获取一元和二元操作符的约束
            una_constraints = [_constraints[op] for op in unary_operators]
            bin_constraints = [_constraints[op] for op in binary_operators]
        else:
            # 如果没有约束，则设置为None
            una_constraints = None
            bin_constraints = None

        # 将嵌套约束字典解析为Julia字典：
        if nested_constraints is not None:
            # 构建Julia字典字符串表示
            nested_constraints_str = "Dict("
            for outer_k, outer_v in nested_constraints.items():
                nested_constraints_str += f"({outer_k}) => Dict("
                for inner_k, inner_v in outer_v.items():
                    nested_constraints_str += f"({inner_k}) => {inner_v}, "
                nested_constraints_str += "), "
            nested_constraints_str += ")"
            # 在Julia中求值该字符串
            nested_constraints = jl.seval(nested_constraints_str)

        # 将操作符复杂度字典解析为Julia字典：
        if complexity_of_operators is not None:
            # 构建Julia字典字符串表示
            complexity_of_operators_str = "Dict("
            for k, v in complexity_of_operators.items():
                complexity_of_operators_str += f"({k}) => {v}, "
            complexity_of_operators_str += ")"
            # 在Julia中求值该字符串
            complexity_of_operators = jl.seval(complexity_of_operators_str)
        # TODO: 将此重构为辅助函数

        # 如果变量复杂度是列表形式，则转换为Julia数组
        if isinstance(complexity_of_variables, list):
            complexity_of_variables = jl_array(complexity_of_variables)

        # 处理自定义损失函数
        custom_loss = jl.seval(
            str(self.elementwise_loss)
            if self.elementwise_loss is not None
            else "nothing"
        )
        # 处理自定义完整目标函数
        custom_full_objective = jl.seval(
            str(self.loss_function) if self.loss_function is not None else "nothing"
        )
        # 处理自定义损失函数表达式
        custom_loss_expression = jl.seval(
            str(self.loss_function_expression)
            if self.loss_function_expression is not None
            else "nothing"
        )

        # 处理早停条件
        early_stop_condition = jl.seval(
            str(self.early_stop_condition)
            if self.early_stop_condition is not None
            else "nothing"
        )

        # 处理输入流
        input_stream = jl.seval(self.input_stream)

        # 加载所需的Julia包
        load_required_packages(
            turbo=self.turbo,
            bumper=self.bumper,
            autodiff_backend=self.autodiff_backend,
            cluster_manager=cluster_manager,
            logger_spec=self.logger_spec,
        )

        # 如果指定了自动微分后端，则设置
        if self.autodiff_backend is not None:
            autodiff_backend = jl.Symbol(self.autodiff_backend)
        else:
            autodiff_backend = None

        # 设置变异权重
        mutation_weights = SymbolicRegression.MutationWeights(
            mutate_constant=self.weight_mutate_constant,          # 常数变异权重
            mutate_operator=self.weight_mutate_operator,         # 操作符变异权重
            swap_operands=self.weight_swap_operands,             # 操作数交换权重
            rotate_tree=self.weight_rotate_tree,                 # 树旋转权重
            add_node=self.weight_add_node,                       # 添加节点权重
            insert_node=self.weight_insert_node,                 # 插入节点权重
            delete_node=self.weight_delete_node,                 # 删除节点权重
            simplify=self.weight_simplify,                       # 简化权重
            randomize=self.weight_randomize,                     # 随机化权重
            do_nothing=self.weight_do_nothing,                   # 无操作权重
            optimize=self.weight_optimize,                       # 优化权重
        )

        # 处理Julia操作符
        jl_binary_operators: list[Any] = []
        jl_unary_operators: list[Any] = []
        # 遍历二元和一元操作符列表
        for input_list, output_list, name in [
            (binary_operators, jl_binary_operators, "binary"),
            (unary_operators, jl_unary_operators, "unary"),
        ]:
            for op in input_list:
                # 在Julia中求值操作符
                jl_op = jl.seval(op)
                # 检查是否为函数
                if not jl_is_function(jl_op):
                    raise ValueError(
                        f"构建`{name}_operators`时，`'{op}'`未返回Julia函数"
                    )
                # 添加到输出列表
                output_list.append(jl_op)

        # 处理复杂度映射
        complexity_mapping = (
            jl.seval(self.complexity_mapping) if self.complexity_mapping else None
        )

        # 处理日志记录器
        if hasattr(self, "logger_") and self.logger_ is not None and self.warm_start:
            logger = self.logger_
        else:
            logger = self.logger_spec.create_logger() if self.logger_spec else None

        # 保存日志记录器引用
        self.logger_ = logger

        # 调用Julia后端。
        # 参见 https://github.com/MilesCranmer/SymbolicRegression.jl/blob/master/src/OptionsStruct.jl
        # 创建符号回归选项结构
        options = SymbolicRegression.Options(
            binary_operators=jl_array(jl_binary_operators, dtype=jl.Function),          # 二元操作符
            unary_operators=jl_array(jl_unary_operators, dtype=jl.Function),            # 一元操作符
            bin_constraints=jl_array(bin_constraints),                                  # 二元约束
            una_constraints=jl_array(una_constraints),                                  # 一元约束
            complexity_of_operators=complexity_of_operators,                            # 操作符复杂度
            complexity_of_constants=self.complexity_of_constants,                       # 常数复杂度
            complexity_of_variables=complexity_of_variables,                            # 变量复杂度
            complexity_mapping=complexity_mapping,                                      # 复杂度映射
            expression_spec=self.expression_spec_.julia_expression_spec(),              # 表达式规范
            nested_constraints=nested_constraints,                                      # 嵌套约束
            elementwise_loss=custom_loss,                                               # 元素级损失函数
            loss_function=custom_full_objective,                                        # 损失函数
            loss_function_expression=custom_loss_expression,                            # 损失函数表达式
            loss_scale=jl.Symbol(self.loss_scale),                                      # 损失缩放
            maxsize=int(self.maxsize),                                                  # 最大大小
            output_directory=_escape_filename(self.output_directory_),                  # 输出目录
            npopulations=int(self.populations),                                         # 种群数量
            batching=self.batching,                                                     # 是否批处理
            batch_size=int(
                min([runtime_params.batch_size, len(X)]) if self.batching else len(X)
            ),                                                                          # 批处理大小
            mutation_weights=mutation_weights,                                          # 变异权重
            tournament_selection_p=self.tournament_selection_p,                         # 锦标赛选择概率
            tournament_selection_n=self.tournament_selection_n,                         # 锦标赛选择数量
            # 这些参数名称相同：
            parsimony=self.parsimony,                                                   # 简约性
            dimensional_constraint_penalty=self.dimensional_constraint_penalty,         # 维度约束惩罚
            dimensionless_constants_only=self.dimensionless_constants_only,             # 仅无量纲常数
            alpha=self.alpha,                                                           # Alpha参数
            maxdepth=runtime_params.maxdepth,                                           # 最大深度
            fast_cycle=self.fast_cycle,                                                 # 快速循环
            turbo=self.turbo,                                                           # Turbo模式
            bumper=self.bumper,                                                         # Bumper模式
            autodiff_backend=autodiff_backend,                                          # 自动微分后端
            migration=self.migration,                                                   # 迁移
            hof_migration=self.hof_migration,                                           # Hall of Fame迁移
            fraction_replaced_hof=self.fraction_replaced_hof,                           # Hall of Fame替换比例
            should_simplify=self.should_simplify,                                       # 是否简化
            should_optimize_constants=self.should_optimize_constants,                   # 是否优化常数
            warmup_maxsize_by=runtime_params.warmup_maxsize_by,                         # 预热最大大小
            use_frequency=self.use_frequency,                                           # 使用频率
            use_frequency_in_tournament=self.use_frequency_in_tournament,               # 锦标赛中使用频率
            adaptive_parsimony_scaling=self.adaptive_parsimony_scaling,                 # 自适应简约性缩放
            npop=self.population_size,                                                  # 种群大小
            ncycles_per_iteration=self.ncycles_per_iteration,                           # 每次迭代的周期数
            fraction_replaced=self.fraction_replaced,                                   # 替换比例
            topn=self.topn,                                                             # Top N
            print_precision=self.print_precision,                                       # 打印精度
            optimizer_algorithm=self.optimizer_algorithm,                               # 优化器算法
            optimizer_nrestarts=self.optimizer_nrestarts,                               # 优化器重启次数
            optimizer_f_calls_limit=self.optimizer_f_calls_limit,                       # 优化器函数调用限制
            optimizer_probability=self.optimize_probability,                            # 优化概率
            optimizer_iterations=self.optimizer_iterations,                             # 优化迭代次数
            perturbation_factor=self.perturbation_factor,                               # 扰动因子
            probability_negate_constant=self.probability_negate_constant,                # 否定常数概率
            annealing=self.annealing,                                                   # 退火
            timeout_in_seconds=self.timeout_in_seconds,                                 # 超时时间（秒）
            crossover_probability=self.crossover_probability,                           # 交叉概率
            skip_mutation_failures=self.skip_mutation_failures,                         # 跳过变异失败
            max_evals=self.max_evals,                                                   # 最大评估次数
            input_stream=input_stream,                                                  # 输入流
            early_stop_condition=early_stop_condition,                                  # 早停条件
            seed=seed,                                                                  # 随机种子
            deterministic=self.deterministic,                                           # 确定性
            define_helper_functions=False,                                              # 不定义辅助函数
        )

        # 序列化选项并保存
        self.julia_options_stream_ = jl_serialize(options)

        # 将数据转换为所需精度
        test_X = np.array(X)
        np_dtype = self._get_precision_mapped_dtype(test_X)

        # 将数据转换为Julia数组：
        jl_X = jl_array(np.array(X, dtype=np_dtype).T)
        # 处理目标值y，根据形状决定是否转置
        if len(y.shape) == 1:
            jl_y = jl_array(np.array(y, dtype=np_dtype))
        else:
            jl_y = jl_array(np.array(y, dtype=np_dtype).T)
        # 处理权重，根据形状决定是否转置
        if weights is not None:
            if len(weights.shape) == 1:
                jl_weights = jl_array(np.array(weights, dtype=np_dtype))
            else:
                jl_weights = jl_array(np.array(weights, dtype=np_dtype).T)
        else:
            jl_weights = None

        # 处理类别数据
        if category is not None:
            offset_for_julia_indexing = 1
            jl_category = jl_array(
                (category + offset_for_julia_indexing).astype(np.int64)
            )
            # 创建包含类别的命名元组
            jl_extra = jl.seval("NamedTuple{(:class,)}")((jl_category,))
        else:
            jl_extra = jl.NamedTuple()

        # 处理多输出情况下的变量名
        if len(y.shape) > 1:
            # 手动设置这些参数以尊重Python的0索引
            # (默认情况下Julia将使用y1, y2...)
            jl_y_variable_names = jl_array(
                [f"y{_subscriptify(i)}" for i in range(y.shape[1])]
            )
        else:
            jl_y_variable_names = None

        # 调用符号回归方程搜索
        out = SymbolicRegression.equation_search(
            jl_X,                          # 输入特征数据
            jl_y,                          # 目标值
            weights=jl_weights,            # 权重
            extra=jl_extra,                # 额外信息（如类别）
            niterations=int(self.niterations),     # 迭代次数
            variable_names=jl_array([str(v) for v in self.feature_names_in_]),      # 变量名
            display_variable_names=jl_array(
                [str(v) for v in self.display_feature_names_in_]
            ),                             # 显示变量名
            y_variable_names=jl_y_variable_names,   # y变量名
            X_units=jl_array(self.X_units_),        # X单位
            y_units=(
                jl_array(self.y_units_)
                if isinstance(self.y_units_, list)
                else self.y_units_
            ),                             # y单位
            options=options,               # 选项
            numprocs=numprocs,             # 进程数
            parallelism=parallelism,       # 并行模式
            saved_state=self.julia_state_, # 保存的状态
            return_state=True,             # 返回状态
            run_id=self.run_id_,           # 运行ID
            addprocs_function=cluster_manager,      # 添加进程函数
            heap_size_hint_in_bytes=self.heap_size_hint_in_bytes,   # 堆大小提示
            progress=runtime_params.progress
            and self.verbosity > 0
            and len(y.shape) == 1,         # 进度显示
            verbosity=int(self.verbosity), # 详细程度
            logger=logger,                 # 日志记录器
        )
        
        # 如果有日志规范，则写入超参数并关闭日志
        if self.logger_spec is not None:
            self.logger_spec.write_hparams(logger, self.get_params())
            if not self.warm_start:
                self.logger_spec.close(logger)

        # 序列化Julia状态并保存
        self.julia_state_stream_ = jl_serialize(out)

        # 设置属性：获取Hall of Fame（最佳方程）
        self.equations_ = self.get_hof(out)

        # 标记已运行
        ALREADY_RAN = True

        # 返回自身
        return self

    def fit(
        self,
        X,
        y,
        *,
        Xresampled=None,
        weights=None,
        variable_names: ArrayLike[str] | None = None,
        complexity_of_variables: int | float | list[int | float] | None = None,
        X_units: ArrayLike[str] | None = None,
        y_units: str | ArrayLike[str] | None = None,
        category: ndarray | None = None,
    ) -> "PySRRegressor":
        """
        搜索适合数据集的方程并将其存储在`self.equations_`中。

        Parameters
        ----------
        X : ndarray | pandas.DataFrame
            训练数据，形状为 (n_samples, n_features)。
        y : ndarray | pandas.DataFrame
            目标值，形状为 (n_samples,) 或 (n_samples, n_targets)。
            如有必要，将被转换为X的数据类型。
        Xresampled : ndarray | pandas.DataFrame
            重采样的训练数据，形状为 (n_resampled, n_features)，
            用于生成去噪数据。这将用作训练数据，而不是`X`。
        weights : ndarray | pandas.DataFrame
            权重数组，形状与`y`相同。
            每个元素表示如何对均方误差损失进行加权
            对于`y`的特定元素。或者，
            如果设置了自定义`loss`，它可以以任意方式使用。
        variable_names : list[str]
            变量名称列表，而不是"x0"、"x1"等。
            如果`X`是pandas数据框，则将使用列名
            而不是`variable_names`。不能包含空格或特殊字符。
            避免使用也是`sympy`中函数名的变量名，如"N"。
        X_units : list[str]
            `X`中每个变量的单位列表。每个单位应该是
            表示Julia表达式的字符串。更多信息请参见DynamicQuantities.jl
            https://symbolicml.org/DynamicQuantities.jl/dev/units/ 。
        y_units : str | list[str]
            与`X_units`类似，但作为目标变量`y`的单位。
            如果`y`是矩阵，则应传递单位列表。如果`X_units`给出但`y_units`未给出，
            则`y_units`将是任意的。
        category : list[int]
            如果`expression_spec`是`ParametricExpressionSpec`，则此参数
            应该是表示每个样本类别的整数列表。

        Returns
        -------
        self : object
            拟合后的估计器对象
        """
        # 初始化在BaseEstimator中未指定的属性
        # 如果启用了warm_start并且已存在julia_state_stream_，则跳过初始化
        if self.warm_start and hasattr(self, "julia_state_stream_"):
            pass
        else:
            # 如果存在julia_state_stream_但没有启用warm_start，则发出警告并重置属性
            if hasattr(self, "julia_state_stream_"):
                warnings.warn(
                    "发现的表达式正在被重置。"
                    "如果您希望继续上次的搜索，请设置`warm_start=True`。",
                )

            # 重置所有相关属性
            self.equations_ = None  # 存储发现的方程
            self.nout_ = 1  # 输出数量，默认为1
            self.selection_mask_ = None  # 特征选择掩码
            self.julia_state_stream_ = None  # Julia状态流
            self.julia_options_stream_ = None  # Julia选项流
            self.complexity_of_variables_ = None  # 变量复杂度
            self.X_units_ = None  # X的单位
            self.y_units_ = None  # y的单位

        # 设置方程文件（用于存储搜索结果）
        self._setup_equation_file()
        # 清空方程文件内容（为新搜索做准备）
        self._clear_equation_file_contents()

        # 验证并修改参数，确保它们符合要求
        runtime_params = self._validate_and_modify_params()

        # 断言检查：如果提供了category，则不能提供Xresampled
        if category is not None:
            assert Xresampled is None

        # 断言检查：如果expression_spec是ParametricExpressionSpec，则必须提供category
        if isinstance(self.expression_spec, ParametricExpressionSpec):
            assert category is not None

        # 验证并设置拟合参数
        (
            X,
            y,
            Xresampled,
            weights,
            variable_names,
            complexity_of_variables,
            X_units,
            y_units,
        ) = self._validate_and_set_fit_params(
            X,
            y,
            Xresampled,
            weights,
            variable_names,
            complexity_of_variables,
            X_units,
            y_units,
        )

        # 如果数据点超过10000个且未启用批处理，则发出警告建议
        if X.shape[0] > 10000 and not self.batching:
            warnings.warn(
                "注意：您正在运行超过10,000个数据点。"
                "您应该考虑开启批处理(batching)(https://ai.damtp.cam.ac.uk/pysr/options/#batching)。"
                "您也应该重新考虑是否需要那么多数据点。"
                "除非您有大量噪声（在这种情况下您应该先平滑数据集），"
                "通常< 10,000个数据点就足以找到符号回归的功能形式。"
                "更多数据点会降低搜索速度。"
            )

        # 检查并设置随机状态，用于numpy随机数生成
        random_state = check_random_state(self.random_state)  # For np random
        # 生成Julia随机数种子
        seed = cast(int, random_state.randint(0, 2**31 - 1))  # For julia random

        # 如果expression_spec是ParametricExpressionSpec，发出参数表达式弃用警告
        if isinstance(self.expression_spec, ParametricExpressionSpec):
            parametric_expression_deprecation_warning(
                self.expression_spec.max_parameters, variable_names
            )

        # 预转换训练数据（特征选择和去噪）
        X, y, variable_names, complexity_of_variables, X_units, y_units = (
            self._pre_transform_training_data(
                X,
                y,
                Xresampled,
                variable_names,
                complexity_of_variables,
                X_units,
                y_units,
                random_state,
            )
        )

        # 设置使用自定义变量名的标志
        use_custom_variable_names = variable_names is not None
        # TODO: 这个条件总是为真

        # 执行各种断言检查，确保数据符合要求
        _check_assertions(
            X,
            use_custom_variable_names,
            variable_names,
            complexity_of_variables,
            weights,
            y,
            X_units,
            y_units,
        )

        # 初始化时保存模型参数，以便在早期退出时可以加载
        if not self.temp_equation_file:
            self._checkpoint()

        # 执行符号回归搜索
        self._run(X, y, runtime_params, weights=weights, seed=seed, category=category)

        # 拟合完成后再次保存，确保pickle文件包含方程
        if not self.temp_equation_file:
            self._checkpoint()

        # 返回拟合后的对象
        return self

    def refresh(self, run_directory: PathLike | None = None) -> None:
        """
        Update self.equations_ with any new options passed.

        For example, updating `extra_sympy_mappings`
        will require a `.refresh()` to update the equations.

        Parameters
        ----------
        checkpoint_file : str or Path
            Path to checkpoint hall of fame file to be loaded.
            The default will use the set `equation_file_`.
        """
        if run_directory is not None:
            self.output_directory_ = str(Path(run_directory).parent)
            self.run_id_ = Path(run_directory).name
            self._clear_equation_file_contents()
        check_is_fitted(self, attributes=["run_id_", "output_directory_"])
        self.equations_ = self.get_hof()

    def predict(
        self,
        X,
        index: int | list[int] | None = None,
        *,
        category: ndarray | None = None,
    ) -> ndarray:
        """
        Predict y from input X using the equation chosen by `model_selection`.

        You may see what equation is used by printing this object. X should
        have the same columns as the training data.

        Parameters
        ----------
        X : ndarray | pandas.DataFrame
            Training data of shape `(n_samples, n_features)`.
        index : int | list[int]
            If you want to compute the output of an expression using a
            particular row of `self.equations_`, you may specify the index here.
            For multiple output equations, you must pass a list of indices
            in the same order.
        category : ndarray | None
            If `expression_spec` is a `ParametricExpressionSpec`, then this
            argument should be a list of integers representing the category
            of each sample in `X`.

        Returns
        -------
        y_predicted : ndarray of shape (n_samples, nout_)
            Values predicted by substituting `X` into the fitted symbolic
            regression model.

        Raises
        ------
        ValueError
            Raises if the `best_equation` cannot be evaluated.
        """
        check_is_fitted(
            self, attributes=["selection_mask_", "feature_names_in_", "nout_"]
        )
        best_equation = self.get_best(index=index)

        # When X is an numpy array or a pandas dataframe with a RangeIndex,
        # the self.feature_names_in_ generated during fit, for the same X,
        # will cause a warning to be thrown during _validate_data.
        # To avoid this, convert X to a dataframe, apply the selection mask,
        # and then set the column/feature_names of X to be equal to those
        # generated during fit.
        if not isinstance(X, pd.DataFrame):
            X = check_array(X)
            X = pd.DataFrame(X)
        if isinstance(X.columns, pd.RangeIndex):
            if self.selection_mask_ is not None:
                # RangeIndex enforces column order allowing columns to
                # be correctly filtered with self.selection_mask_
                X = X[X.columns[self.selection_mask_]]
            X.columns = self.feature_names_in_
        # Without feature information, CallableEquation/lambda_format equations
        # require that the column order of X matches that of the X used during
        # the fitting process. _validate_data removes this feature information
        # when it converts the dataframe to an np array. Thus, to ensure feature
        # order is preserved after conversion, the dataframe columns must be
        # reordered/reindexed to match those of the transformed (denoised and
        # feature selected) X in fit.
        X = X.reindex(columns=self.feature_names_in_)
        X = self._validate_data_X(X)
        if self.expression_spec_.evaluates_in_julia:
            # Julia wants the right dtype
            X = X.astype(self._get_precision_mapped_dtype(X))

        if category is not None:
            offset_for_julia_indexing = 1
            args: tuple = (
                jl_array((category + offset_for_julia_indexing).astype(np.int64)),
            )
        else:
            args = ()

        try:
            if isinstance(best_equation, list):
                assert self.nout_ > 1
                return np.stack(
                    [
                        cast(ndarray, eq["lambda_format"](X, *args))
                        for eq in best_equation
                    ],
                    axis=1,
                )
            else:
                return cast(ndarray, best_equation["lambda_format"](X, *args))
        except Exception as error:
            raise ValueError(
                "Failed to evaluate the expression. "
                "If you are using a custom operator, make sure to define it in `extra_sympy_mappings`, "
                "e.g., `model.set_params(extra_sympy_mappings={'inv': lambda x: 1/x})`, where "
                "`lambda x: 1/x` is a valid SymPy function defining the operator. "
                "You can then run `model.refresh()` to re-load the expressions."
            ) from error

    def sympy(self, index: int | list[int] | None = None):
        """
        Return sympy representation of the equation(s) chosen by `model_selection`.

        Parameters
        ----------
        index : int | list[int]
            If you wish to select a particular equation from
            `self.equations_`, give the index number here. This overrides
            the `model_selection` parameter. If there are multiple output
            features, then pass a list of indices with the order the same
            as the output feature.

        Returns
        -------
        best_equation : str, list[str] of length nout_
            SymPy representation of the best equation.
        """
        if not self.expression_spec_.supports_sympy:
            raise ValueError(
                f"`expression_spec={self.expression_spec_}` does not support sympy export."
            )
        self.refresh()
        best_equation = self.get_best(index=index)
        if isinstance(best_equation, list):
            assert self.nout_ > 1
            return [eq["sympy_format"] for eq in best_equation]
        else:
            return best_equation["sympy_format"]

    def latex(
        self, index: int | list[int] | None = None, precision: int = 3
    ) -> str | list[str]:
        """
        Return latex representation of the equation(s) chosen by `model_selection`.

        Parameters
        ----------
        index : int | list[int]
            If you wish to select a particular equation from
            `self.equations_`, give the index number here. This overrides
            the `model_selection` parameter. If there are multiple output
            features, then pass a list of indices with the order the same
            as the output feature.
        precision : int
            The number of significant figures shown in the LaTeX
            representation.
            Default is `3`.

        Returns
        -------
        best_equation : str or list[str] of length nout_
            LaTeX expression of the best equation.
        """
        if not self.expression_spec_.supports_latex:
            raise ValueError(
                f"`expression_spec={self.expression_spec_}` does not support latex export."
            )
        self.refresh()
        sympy_representation = self.sympy(index=index)
        if self.nout_ > 1:
            output = []
            for s in sympy_representation:
                latex = sympy2latex(s, prec=precision)
                output.append(latex)
            return output
        return sympy2latex(sympy_representation, prec=precision)

    def jax(self, index=None):
        """
        Return jax representation of the equation(s) chosen by `model_selection`.

        Each equation (multiple given if there are multiple outputs) is a dictionary
        containing {"callable": func, "parameters": params}. To call `func`, pass
        func(X, params). This function is differentiable using `jax.grad`.

        Parameters
        ----------
        index : int | list[int]
            If you wish to select a particular equation from
            `self.equations_`, give the index number here. This overrides
            the `model_selection` parameter. If there are multiple output
            features, then pass a list of indices with the order the same
            as the output feature.

        Returns
        -------
        best_equation : dict[str, Any]
            Dictionary of callable jax function in "callable" key,
            and jax array of parameters as "parameters" key.
        """
        if not self.expression_spec_.supports_jax:
            raise ValueError(
                f"`expression_spec={self.expression_spec_}` does not support jax export."
            )
        self.set_params(output_jax_format=True)
        self.refresh()
        best_equation = self.get_best(index=index)
        if isinstance(best_equation, list):
            assert self.nout_ > 1
            return [eq["jax_format"] for eq in best_equation]
        else:
            return best_equation["jax_format"]

    def pytorch(self, index=None):
        """
        Return pytorch representation of the equation(s) chosen by `model_selection`.

        Each equation (multiple given if there are multiple outputs) is a PyTorch module
        containing the parameters as trainable attributes. You can use the module like
        any other PyTorch module: `module(X)`, where `X` is a tensor with the same
        column ordering as trained with.

        Parameters
        ----------
        index : int | list[int]
            If you wish to select a particular equation from
            `self.equations_`, give the index number here. This overrides
            the `model_selection` parameter. If there are multiple output
            features, then pass a list of indices with the order the same
            as the output feature.

        Returns
        -------
        best_equation : torch.nn.Module
            PyTorch module representing the expression.
        """
        if not self.expression_spec_.supports_torch:
            raise ValueError(
                f"`expression_spec={self.expression_spec_}` does not support torch export."
            )
        self.set_params(output_torch_format=True)
        self.refresh()
        best_equation = self.get_best(index=index)
        if isinstance(best_equation, list):
            return [eq["torch_format"] for eq in best_equation]
        else:
            return best_equation["torch_format"]

    def get_equation_file(self, i: int | None = None) -> Path:
        if i is not None:
            return (
                Path(self.output_directory_)
                / self.run_id_
                / f"hall_of_fame_output{i}.csv"
            )
        else:
            return Path(self.output_directory_) / self.run_id_ / "hall_of_fame.csv"

    def _read_equation_file(self) -> list[pd.DataFrame]:
        """Read the hall of fame file created by `SymbolicRegression.jl`."""

        try:
            if self.nout_ > 1:
                all_outputs = []
                for i in range(1, self.nout_ + 1):
                    cur_filename = str(self.get_equation_file(i)) + ".bak"
                    if not os.path.exists(cur_filename):
                        cur_filename = str(self.get_equation_file(i))
                    with open(cur_filename, "r", encoding="utf-8") as f:
                        buf = f.read()
                    buf = _preprocess_julia_floats(buf)
                    df = self._postprocess_dataframe(pd.read_csv(StringIO(buf)))
                    all_outputs.append(df)
            else:
                filename = str(self.get_equation_file()) + ".bak"
                if not os.path.exists(filename):
                    filename = str(self.get_equation_file())
                with open(filename, "r", encoding="utf-8") as f:
                    buf = f.read()
                buf = _preprocess_julia_floats(buf)
                all_outputs = [self._postprocess_dataframe(pd.read_csv(StringIO(buf)))]

        except FileNotFoundError:
            raise RuntimeError(
                "Couldn't find equation file! The equation search likely exited "
                "before a single iteration completed."
            )
        return all_outputs

    def _postprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(
            columns={
                "Complexity": "complexity",
                "Loss": "loss",
                "Equation": "equation",
            },
        )

        return df

    def get_hof(self, search_output=None) -> pd.DataFrame | list[pd.DataFrame]:
        """Get the equations from a hall of fame file or search output.

        If no arguments entered, the ones used
        previously from a call to PySR will be used.
        """
        check_is_fitted(
            self,
            attributes=[
                "nout_",
                "run_id_",
                "output_directory_",
                "selection_mask_",
                "feature_names_in_",
            ],
        )
        should_read_from_file = (
            not hasattr(self, "equation_file_contents_")
            or self.equation_file_contents_ is None
        )
        if should_read_from_file:
            self.equation_file_contents_ = self._read_equation_file()

        _validate_export_mappings(self.extra_jax_mappings, self.extra_torch_mappings)

        equation_file_contents = cast(List[pd.DataFrame], self.equation_file_contents_)

        ret_outputs = [
            pd.concat(
                [
                    output,
                    *([calculate_scores(output)] if self.loss_scale == "log" else []),
                    self.expression_spec_.create_exports(
                        self, output, search_output, i if self.nout_ > 1 else None
                    ),
                ],
                axis=1,
            )
            for i, output in enumerate(equation_file_contents)
        ]

        if self.nout_ > 1:
            return ret_outputs
        return ret_outputs[0]

    def latex_table(
        self,
        indices: list[int] | None = None,
        precision: int = 3,
        columns: list[str] = ["equation", "complexity", "loss", "score"],
    ) -> str:
        """Create a LaTeX/booktabs table for all, or some, of the equations.

        Parameters
        ----------
        indices : list[int] | list[list[int]]
            If you wish to select a particular subset of equations from
            `self.equations_`, give the row numbers here. By default,
            all equations will be used. If there are multiple output
            features, then pass a list of lists.
        precision : int
            The number of significant figures shown in the LaTeX
            representations.
            Default is `3`.
        columns : list[str]
            Which columns to include in the table.
            Default is `["equation", "complexity", "loss", "score"]`.

        Returns
        -------
        latex_table_str : str
            A string that will render a table in LaTeX of the equations.
        """
        if not self.expression_spec_.supports_latex:
            raise ValueError(
                f"`expression_spec={self.expression_spec_}` does not support latex export."
            )
        self.refresh()

        if isinstance(self.equations_, list):
            if indices is not None:
                assert isinstance(indices, list)
                assert isinstance(indices[0], list)
                assert len(indices) == self.nout_

            table_string = sympy2multilatextable(
                self.equations_, indices=indices, precision=precision, columns=columns
            )
        elif isinstance(self.equations_, pd.DataFrame):
            if indices is not None:
                assert isinstance(indices, list)
                assert isinstance(indices[0], int)

            table_string = sympy2latextable(
                self.equations_, indices=indices, precision=precision, columns=columns
            )
        else:
            raise ValueError(
                "Invalid type for equations_ to pass to `latex_table`. "
                "Expected a DataFrame or a list of DataFrames."
            )

        return with_preamble(table_string)


def idx_model_selection(equations: pd.DataFrame, model_selection: str):
    """Select an expression and return its index."""

    # We must default to "accuracy" if no score column is present (like in the case of linear loss_scale)
    model_selection = model_selection if "score" in equations.columns else "accuracy"

    if model_selection == "accuracy":
        chosen_idx = equations["loss"].idxmin()
    elif model_selection == "best":
        threshold = 1.5 * equations["loss"].min()
        filtered_equations = equations.query(f"loss <= {threshold}")
        chosen_idx = filtered_equations["score"].idxmax()
    elif model_selection == "score":
        chosen_idx = equations["score"].idxmax()
    else:
        raise NotImplementedError(
            f"{model_selection} is not a valid model selection strategy."
        )
    return chosen_idx


def calculate_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate scores for each equation based on loss and complexity.

    Score is defined as the negated derivative of the log-loss with respect to complexity.
    A higher score means the equation achieved a much better loss at a slightly higher complexity.
    """
    scores = []
    lastMSE = None
    lastComplexity = 0

    for _, row in df.iterrows():
        curMSE = row["loss"]
        curComplexity = row["complexity"]

        if lastMSE is None:
            cur_score = 0.0
        else:
            if curMSE > 0.0:
                cur_score = -np.log(curMSE / lastMSE) / (curComplexity - lastComplexity)
            else:
                cur_score = np.inf

        scores.append(cur_score)
        lastMSE = curMSE
        lastComplexity = curComplexity

    return pd.DataFrame(
        {
            "score": np.array(scores),
        },
        index=df.index,
    )


def _mutate_parameter(param_name: str, param_value):
    if param_name == "batch_size" and param_value < 1:
        warnings.warn(
            "Given `batch_size` must be greater than or equal to one. "
            "`batch_size` has been increased to equal one."
        )
        return 1

    if (
        param_name == "progress"
        and param_value == True
        and "buffer" not in sys.stdout.__dir__()
    ):
        warnings.warn(
            "Note: it looks like you are running in Jupyter. "
            "The progress bar will be turned off."
        )
        return False

    return param_value


def _map_parallelism_params(
    parallelism: Literal["serial", "multithreading", "multiprocessing"] | None,
    procs: int | None,
    multithreading: bool | None,
) -> tuple[Literal["serial", "multithreading", "multiprocessing"], int | None]:
    """Map old and new parallelism parameters to the new format.

    Parameters
    ----------
    parallelism : str or None
        New parallelism parameter. Can be "serial", "multithreading", or "multiprocessing".
    procs : int or None
        Number of processes parameter.
    multithreading : bool or None
        Old multithreading parameter.

    Returns
    -------
    parallelism : str
        Mapped parallelism mode.
    procs : int or None
        Mapped number of processes.

    Raises
    ------
    ValueError
        If both old and new parameters are specified, or if invalid combinations are given.
    """
    # Check for mixing old and new parameters
    using_new = parallelism is not None
    using_old = multithreading is not None

    if using_new and using_old:
        raise ValueError(
            "Cannot mix old and new parallelism parameters. "
            "Use either `parallelism` and `numprocs`, or `procs` and `multithreading`."
        )
    elif using_old:
        warnings.warn(
            "The `multithreading: bool` parameter has been deprecated in favor "
            "of `parallelism: Literal['multithreading', 'serial', 'multiprocessing']`.\n"
            "Previous usage of `multithreading=True` (default) is now `parallelism='multithreading'`; "
            "`multithreading=False, procs=0` is now `parallelism='serial'`; and "
            "`multithreading=True, procs={int}` is now `parallelism='multiprocessing', procs={int}`."
        )
        if multithreading:
            _parallelism: Literal["multithreading", "multiprocessing", "serial"] = (
                "multithreading"
            )
            _procs = None
        elif procs is not None and procs > 0:
            _parallelism = "multiprocessing"
            _procs = procs
        else:
            _parallelism = "serial"
            _procs = None
    elif using_new:
        _parallelism = cast(
            Literal["serial", "multithreading", "multiprocessing"], parallelism
        )
        _procs = procs
    else:
        _parallelism = "multithreading"
        _procs = None

    if _parallelism not in {"serial", "multithreading", "multiprocessing"}:
        raise ValueError(
            "`parallelism` must be one of 'serial', 'multithreading', or 'multiprocessing'"
        )
    elif _parallelism == "serial" and _procs is not None:
        warnings.warn(
            "`numprocs` is specified but will be ignored since `parallelism='serial'`"
        )
        _procs = None
    elif parallelism == "multithreading" and _procs is not None:
        warnings.warn(
            "`numprocs` is specified but will be ignored since `parallelism='multithreading'`"
        )
        _procs = None
    elif parallelism == "multiprocessing" and _procs is None:
        _procs = cpu_count()

    return _parallelism, _procs
