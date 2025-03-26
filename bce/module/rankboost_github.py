# from https://github.com/pcoving/KDDCup/blob/master/train.py
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble.base import BaseEnsemble
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor

from itertools import tee, izip
import sys

class Stump(BaseEstimator):
    """决策树桩分类器,作为弱学习器"""

    def __init__(self):
        self.featureidx = None  # 特征索引
        self.splitval = None    # 分割值
        self.above = None       # 是否大于分割值

    def pairwise(self, iterable):
        """生成迭代器的相邻元素对
        
        Args:
            iterable: 输入的迭代器
        Returns:
            相邻元素对的迭代器
        """
        a, b = tee(iterable)
        next(b, None)
        return izip(a, b)

    def fit(self, X, y, X_argsorted=None, sample_weight=None, npositive=None):
        """训练决策树桩
        
        Args:
            X: 特征矩阵
            y: 标签
            X_argsorted: 按特征值排序的索引矩阵
            sample_weight: 样本权重
            npositive: 正样本数量
        """
        # 复制样本权重并对负样本取反
        mysample_weight = np.copy(sample_weight)
        mysample_weight[npositive:] *= -1

        mysample_weight_sum = np.sum(mysample_weight)

        # 选择最优分割点,使得E+[h(x+)] - E-[h(x-)]最大化
        max_val = 0.0
        for fidx in range(X.shape[1]):
            val = 0.0
            # 遍历排序后的特征值对
            for xidx, xidx2 in self.pairwise(X_argsorted[:,fidx]):
                val += mysample_weight[xidx]
                # 只在不同值之间分割
                if X[xidx, fidx] != X[xidx2, fidx]:
                    if val > max_val:
                        max_val = val
                        self.featureidx = fidx
                        self.splitval = X[xidx, fidx]
                        self.above = False
                    if (mysample_weight_sum-val) > max_val:
                        max_val = mysample_weight_sum-val
                        self.featureidx = fidx
                        self.splitval = X[xidx, fidx]
                        self.above = True

    def predict(self, X):
        """预测样本类别

        Args:
            X: 特征矩阵
        Returns:
            预测的类别标签
        """
        if self.above:
            y = [ int(x[self.featureidx] > self.splitval) for x in X ]
        else:
            y = [ int(x[self.featureidx] <= self.splitval) for x in X ]

        return np.asarray(y)

class BipartiteRankBoost(BaseEnsemble):
    """二分类排序提升算法"""

    def __init__(self,
                 n_estimators=50,
                 learning_rate=1.,
                 verbose=0,
                 base_estimator=Stump()):
        """初始化

        Args:
            n_estimators: 弱学习器数量
            learning_rate: 学习率
            verbose: 日志级别
            base_estimator: 基础弱学习器
        """
        if base_estimator==None:
            base_estimator=Stump()
        super(BipartiteRankBoost, self).__init__(
            base_estimator=Stump(),
            n_estimators=n_estimators,
            estimator_params=tuple())

        self.base_estimator = base_estimator
        self.classes_ = [0, 1]
        self.base_estimator_ = base_estimator
        self.estimator_weights_ = None
        self.learning_rate = learning_rate
        self.verbose = verbose

    def fit(self, X, y):
        """训练模型
        
        Args:
            X: 特征矩阵
            y: 标签
        Returns:
            训练好的模型
        """
        X, y = np.array(X), np.array(y)

        # 确保标签为0/1
        assert np.max(y) == 1
        assert np.min(y) == 0

        # 按标签排序,正样本在前
        sortidx = y.argsort()[::-1]
        X, y = X[sortidx], y[sortidx]

        # 计算正样本数量
        npositive = sum([1 for val in y if val == 1])

        # 初始化样本权重
        sample_weight = np.empty(X.shape[0], dtype=np.float)
        sample_weight[:npositive] = 1. / npositive  # 正样本权重
        sample_weight[npositive:] = 1. / (X.shape[0]-npositive)

        self.estimators_ = []
        self.estimator_weights_ =  np.zeros(self.n_estimators, dtype=np.float)

        # 创建特征值排序索引用于快速训练
        X_argsorted = np.asfortranarray(np.argsort(X.T, axis=1).astype(np.int32).T)

        for iboost in range(self.n_estimators):
            sample_weight, estimator_weight = self._boost(iboost,
                                                          X, y,
                                                          sample_weight,
                                                          X_argsorted=X_argsorted,
                                                          npositive=npositive)

            self.estimator_weights_[iboost] = estimator_weight

            # 重新归一化样本权重
            sample_weight[:npositive] /= np.sum(sample_weight[:npositive])
            sample_weight[npositive:] /= np.sum(sample_weight[npositive:])

        return self

    def _boost(self, iboost, X, y, sample_weight, X_argsorted, npositive):
        """训练单个弱学习器

        Args:
            iboost: 当前迭代次数
            X: 特征矩阵
            y: 标签
            sample_weight: 样本权重
            X_argsorted: 特征值排序索引
            npositive: 正样本数量
        Returns:
            更新后的样本权重和弱学习器权重
        """
        estimator = self._make_estimator()

        if self.verbose == 2:
            print('building stump {} out of {}'.format(iboost+1, self.n_estimators))
        elif self.verbose == 1:
            sys.stdout.write('.')
            sys.stdout.flush()
        estimator.fit(X, y, sample_weight=sample_weight,
                      X_argsorted=X_argsorted, npositive=npositive)

        y_predict = estimator.predict(X)

        # 计算正负样本的加权错误率
        positive_error = np.average(y_predict[:npositive], weights=sample_weight[:npositive], axis=0)
        negative_error = np.average(y_predict[npositive:], weights=sample_weight[npositive:], axis=0)

        # 计算弱学习器权重
        estimator_weight = self.learning_rate*np.log(positive_error/negative_error)/2.

        # 更新样本权重
        sample_weight[:npositive] *= np.exp(-estimator_weight*y_predict[:npositive])
        sample_weight[npositive:] *= np.exp(estimator_weight*y_predict[npositive:])

        return sample_weight, estimator_weight

    def predict_proba(self, X):
        """预测样本的概率分数
        
        Args:
            X: 特征矩阵
        Returns:
            预测的概率分数
        """
        X = np.array(X)

        y_predict = np.zeros(X.shape[0])

        # 计算加权预测分数
        for weight, estimator in zip(self.estimator_weights_, self.estimators_):
            y_predict += estimator.predict(X)*weight
        y_predict /= sum(self.estimator_weights_)

        # 输出两列相同的分数以兼容其他分类器
        return np.asarray([y_predict, y_predict]).T
