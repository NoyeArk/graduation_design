import torch
import numpy as np
from sklearn.tree import DecisionTreeRegressor


class GBRank:
    def __init__(self, args, n_items):
        self.device = args.device
        self.n_estimators = args.n_estimators
        self.learning_rate = args.lr
        self.threshold = args.threshold
        self.trees = []
        self.n_items = n_items

    def fit(self, batch):
        """
        训练模型
        """
        # 获取所有唯一的项目ID
        all_items = torch.unique(torch.cat([
            batch['user_seq'].flatten(),
            batch['pos_items'],
            batch['neg_items']
        ])).cpu().numpy()

        # 创建项目ID到索引的映射
        self.item_to_idx = {item: idx for idx, item in enumerate(all_items)}
        n_items = len(all_items)

        # 将序列和项目ID转换为索引
        user_seq = np.array([[self.item_to_idx[item.item()] for item in seq] 
                            for seq in batch['user_seq'].cpu()])
        pos_items = np.array([self.item_to_idx[item.item()] 
                            for item in batch['pos_items'].cpu()])
        neg_items = np.array([self.item_to_idx[item.item()] 
                            for item in batch['neg_items'].cpu()])
        
        # 初始化预测值
        f_prev = np.zeros(n_items)
        
        # 迭代训练决策树
        for i in range(self.n_estimators):
            # 创建训练数据
            X_train, y_train = self._create_training_data(user_seq, f_prev, pos_items, neg_items)
            
            if len(X_train) == 0:
                print(f"Iteration {i}: No violations found, stopping early.")
                break
                
            # 训练新的决策树
            tree = DecisionTreeRegressor(max_depth=3)
            tree.fit(X_train, y_train)
            self.trees.append(tree)
            
            # 更新预测值
            predictions = tree.predict(np.arange(n_items).reshape(-1, 1))
            f_prev += self.learning_rate * predictions
            
            # 打印训练进度
            violations = self._count_violations(f_prev, pos_items, neg_items)
            print(f"Iteration {i+1}: {violations} violations")
    
    def _create_training_data(self, X, f_prev, pos_items, neg_items):
        """
        创建训练数据
        """
        X_train = []
        y_train = []
        
        for i in range(len(pos_items)):
            # 计算正负样本的预测值差异
            pos_diff = f_prev[pos_items[i]] - f_prev[neg_items[i]]
            
            # 如果违反排序约束，添加到训练数据
            if pos_diff < self.threshold:
                # 添加正样本
                X_train.append([pos_items[i]])
                y_train.append(f_prev[neg_items[i]] + self.threshold)
                
                # 添加负样本
                X_train.append([neg_items[i]])
                y_train.append(f_prev[pos_items[i]] - self.threshold)
        
        if len(X_train) > 0:
            return np.array(X_train), np.array(y_train)
        return np.array([]).reshape(0, 1), np.array([])
    
    def _count_violations(self, scores, pos_items, neg_items):
        """
        计算违反排序约束的数量
        """
        violations = 0
        for i in range(len(pos_items)):
            if scores[pos_items[i]] - scores[neg_items[i]] < self.threshold:
                violations += 1
        return violations
    
    def predict(self, batch):
        """
        预测排序分数
        """
        if not self.trees:
            raise ValueError("Model not trained yet. Call fit() first.")

        # 将输入转换为索引
        indices = np.array([[self.item_to_idx.get(item.item(), 0) for item in seq]
                          for seq in batch['user_seq'].cpu()])

        # 初始化分数
        scores = np.zeros(len(self.item_to_idx))

        # 累加所有树的预测值
        for tree in self.trees:
            scores += self.learning_rate * tree.predict(np.arange(len(self.item_to_idx)).reshape(-1, 1))

        # 获取序列中每个位置的分数
        batch_scores = scores[indices]
        return torch.FloatTensor(batch_scores).to(self.device)

# 测试代码
def test_gbrank():
    class Args:
        def __init__(self):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.n_estimators = 10
            self.lr = 0.001
            self.threshold = 0.1
    
    args = Args()
    model = GBRank(args)
    
    # 创建模拟数据
    batch_size = 1
    seq_len = 10
    n_items = 100
    
    batch = {
        'user_seq': torch.randint(0, n_items, (batch_size, seq_len)).to(args.device),
        'pos_items': torch.randint(0, n_items, (batch_size,)).to(args.device),
        'neg_items': torch.randint(0, n_items, (batch_size,)).to(args.device)
    }
    
    # 训练模型
    for i in range(1):
        model.fit(batch)
    
    # 预测
    scores = model.predict(batch)
    print(scores.shape)
    print("Prediction shape:", scores.shape)
    print("Sample predictions:", scores[:5])

if __name__ == "__main__":
    test_gbrank()
