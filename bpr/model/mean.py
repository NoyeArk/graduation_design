import torch
import numpy as np
from sklearn.tree import DecisionTreeRegressor


class Stack:
    def __init__(self, args, n_items):
        """
        初始化Stack模型
        
        Args:
            args: 参数配置
            n_items: 物品总数
        """
        self.device = args.device
        self.n_estimators = args.n_estimators
        self.learning_rate = args.lr
        self.trees = []
        self.n_items = n_items

    def fit(self, batch):
        """
        训练模型
        
        Args:
            batch: 包含训练数据的字典
                user_seq: 用户历史序列 [batch_size, seq_len]
                pos_items: 正样本 [batch_size]
                neg_items: 负样本 [batch_size] 
                base_model_preds: 基模型预测值 [batch_size, k, n_items]
        """
        # 获取基模型数量
        n_models = batch['base_model_preds'].shape[1]
        
        # 将序列和项目ID转换为numpy数组
        pos_items = batch['pos_items'].cpu().numpy()
        neg_items = batch['neg_items'].cpu().numpy()
        base_preds = batch['base_model_preds'].cpu().numpy()
        
        # 为每个基模型创建训练数据
        X_train = []
        y_train = []
        
        for i in range(len(pos_items)):
            # 获取每个基模型对正负样本的预测分数
            pos_scores = base_preds[i, :, pos_items[i]]  # [k]
            neg_scores = base_preds[i, :, neg_items[i]]  # [k]
            
            # 添加到训练数据
            X_train.append(pos_scores)
            X_train.append(neg_scores)
            y_train.append(1.0)  # 正样本标签
            y_train.append(0.0)  # 负样本标签
            
        X_train = np.array(X_train)  # [2*batch_size, k]
        y_train = np.array(y_train)  # [2*batch_size]
        
        # 训练决策树
        tree = DecisionTreeRegressor(max_depth=3)
        tree.fit(X_train, y_train)
        self.trees.append(tree)
        
    def predict(self, batch):
        """
        预测排序分数
        
        Args:
            batch: 包含预测数据的字典
                base_model_preds: 基模型预测值 [batch_size, k, n_items]
                
        Returns:
            scores: 最终的预测分数 [batch_size, seq_len]
        """
        if not self.trees:
            raise ValueError("模型未训练,请先调用fit()方法")

        # 获取基模型预测值
        base_preds = batch['base_model_preds']
        batch_size, n_models, n_items = base_preds.shape

        # 重塑预测值用于决策树输入
        base_preds = base_preds.reshape(-1, n_models).cpu().numpy()
        
        # 获取决策树预测
        scores = np.zeros(len(base_preds))
        for tree in self.trees:
            scores += self.learning_rate * tree.predict(base_preds)
            
        # 重塑回原始形状
        scores = scores.reshape(batch_size, -1)
        
        return torch.FloatTensor(scores).to(self.device)
