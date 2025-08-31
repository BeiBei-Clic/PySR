"""Functions for doing feature selection during preprocessing."""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
from numpy import ndarray
from numpy.typing import NDArray

from .utils import ArrayLike

pysr_logger = logging.getLogger(__name__)


def run_feature_selection(
    X: ndarray,
    y: ndarray,
    select_k_features: int,
    random_state: np.random.RandomState | None = None,
) -> NDArray[np.bool_]:
    """
    查找最重要的特征。

    使用梯度提升树回归器作为代理来查找
    X中最重要的k个特征，返回这些特征的索引作为输出。

    Parameters
    ----------
    X : ndarray
        输入特征数据，形状为(n_samples, n_features)
    y : ndarray
        目标值，形状为(n_samples,)或(n_samples, n_targets)
    select_k_features : int
        要选择的特征数量
    random_state : np.random.RandomState | None
        随机数生成器状态，用于确保结果可重现

    Returns
    -------
    NDArray[np.bool_]
        布尔数组，表示每个特征是否被选中
    """
    # 从sklearn导入随机森林回归器和特征选择器
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import SelectFromModel

    # 创建随机森林回归器作为特征重要性评估器
    # 使用100个决策树，最大深度为3，设置随机状态以确保结果可重现
    clf = RandomForestRegressor(
        n_estimators=100, max_depth=3, random_state=random_state
    )
    
    # 使用输入数据训练随机森林模型
    clf.fit(X, y)
    
    # 创建特征选择器，从训练好的模型中选择特征
    # threshold=-np.inf 表示不设置阈值下限，选择所有特征
    # max_features=select_k_features 限制最多选择的特征数量
    # prefit=True 表示模型已经训练好了，不需要重新训练
    selector = SelectFromModel(
        clf, threshold=-np.inf, max_features=select_k_features, prefit=True
    )
    
    # 获取特征选择结果，返回布尔数组（indices=False）
    # 数组中为True的位置表示对应的特征被选中
    return cast(NDArray[np.bool_], selector.get_support(indices=False))