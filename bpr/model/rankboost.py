import torch
import torch.nn as nn
import torch.nn.functional as F


class RankBoost(nn.Module):
    def __init__(self, args, n_items):
        super(RankBoost, self).__init__()
        self.device = args['device']
        self.n_items = n_items
        self.dim = args['hidden_dim']
        self.n_weak_learners = args['n_weak_learners']  # 弱学习器数量
        self.learning_rate = args['lr']

        # 初始化弱分类器的权重
        self.weak_learners = nn.ModuleList([
            WeakLearner(self.dim) for _ in range(self.n_weak_learners)
        ])

        # 初始化弱分类器的组合系数 alpha
        self.alphas = nn.Parameter(torch.ones(self.n_weak_learners) / self.n_weak_learners)

        # 序列编码器
        self.seq_encoder = nn.LSTM(
            input_size=self.dim,
            hidden_size=self.dim,
            batch_first=True
        )
        self.item_embeddings = nn.Embedding(self.n_items + 1, self.dim, padding_idx=0)
        self.to(self.device)

    def forward(self, batch):
        """
        前向传播函数,用于训练和预测
        
        Args:
            batch: 包含用户序列和标签的字典
            
        Returns:
            final_scores: 所有项目的预测得分
            loss: 训练时的损失值(预测时为None)
        """
        user_seq = batch['user_seq']
        pos_items = batch.get('pos_item')
        neg_items = batch.get('neg_item')

        # 获取序列表示
        seq_emb = self.item_embeddings(user_seq)  # [batch_size, seq_len, dim]
        seq_output, _ = self.seq_encoder(seq_emb)  # [batch_size, seq_len, dim]
        seq_repr = seq_output[:, -1, :]  # 取最后一个时间步的输出 [batch_size, dim]

        # 获取候选项目的嵌入
        all_items = torch.arange(self.n_items, device=self.device)
        item_emb = self.item_embeddings(all_items)  # [n_items, dim]

        # 计算每个弱学习器的得分
        weak_scores = []
        for learner in self.weak_learners:
            scores = learner(seq_repr, item_emb)  # [batch_size, n_items]
            weak_scores.append(scores)

        # 组合弱学习器的得分
        final_scores = torch.zeros_like(weak_scores[0])
        for alpha, scores in zip(self.alphas, weak_scores):
            final_scores += alpha * scores

        loss = self.rankboost_loss(final_scores, pos_items, neg_items)

        return final_scores, loss

    def rankboost_loss(self, scores, pos_items, neg_items):
        """
        计算 RankBoost 损失函数
        """
        # 获取正样本和负样本的得分
        pos_scores = torch.gather(scores, 1, pos_items.unsqueeze(1))  # [batch_size, 1]
        neg_scores = torch.gather(scores, 1, neg_items.unsqueeze(1))  # [batch_size, 1]

        # 计算排序损失
        diff = pos_scores - neg_scores  # [batch_size, 1]
        loss = torch.mean(torch.exp(-diff))  # 指数损失函数

        return loss

    def update_alphas(self, weak_scores, pos_items, neg_items, weights):
        """
        更新弱分类器的组合系数
        
        Args:
            weak_scores: 每个弱分类器的预测得分列表
            pos_items: 正样本项目ID
            neg_items: 负样本项目ID  
            weights: 样本权重
        """
        with torch.no_grad():
            # 计算每个弱分类器的错误率
            errors = []
            for scores in weak_scores:
                pos_scores = torch.gather(scores, 1, pos_items.unsqueeze(1))
                neg_scores = torch.gather(scores, 1, neg_items)
                diff = pos_scores - neg_scores
                error = torch.sum(weights * (diff < 0).float())
                errors.append(error)

            # 更新 alpha 值
            errors = torch.stack(errors)
            self.alphas.data = 0.5 * torch.log((1 - errors) / errors)

            # 归一化
            self.alphas.data = F.softmax(self.alphas.data, dim=0)


class WeakLearner(nn.Module):
    """
    弱学习器实现
    """
    def __init__(self, dim):
        super(WeakLearner, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, seq_repr, item_emb):
        """
        Args:
            seq_repr: [batch_size, dim] 序列表示
            item_emb: [n_items, dim] 所有项目的嵌入
        Returns:
            scores: [batch_size, n_items] 排序得分
        """
        # 计算序列表示与所有项目的相似度
        batch_size = seq_repr.size(0)
        n_items = item_emb.size(0)

        # 扩展维度以便计算
        seq_repr = seq_repr.unsqueeze(1).expand(-1, n_items, -1)  # [batch_size, n_items, dim]
        item_emb = item_emb.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, n_items, dim]

        # 连接序列表示和项目嵌入
        concat = torch.cat([seq_repr, item_emb], dim=-1)  # [batch_size, n_items, 2*dim]

        # 投影得到得分
        scores = self.projection(concat).squeeze(-1)  # [batch_size, n_items]

        return scores


def rank_items(model, user_seq, candidate_items=None):
    """
    使用 RankBoost 模型对物品进行排序
    
    Args:
        model: 训练好的 RankBoost 模型
        user_seq: 用户的历史交互序列 [batch_size, seq_len]
        candidate_items: 候选物品列表，如果为 None 则对所有物品排序
        
    Returns:
        ranked_items: 排序后的物品 ID 列表 [batch_size, n_items]
        scores: 对应的预测分数 [batch_size, n_items]
    """
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        # 准备输入数据
        user_seq = torch.LongTensor(user_seq)  # [batch_size, seq_len]
        user_seq = user_seq.to(model.device)
        
        # 如果没有指定候选物品，则使用所有物品
        if candidate_items is None:
            candidate_items = torch.arange(model.n_items, device=model.device)
        else:
            candidate_items = torch.LongTensor(candidate_items).to(model.device)
        
        # 构造批次数据
        batch = {
            'user_seq': user_seq,
            'pos_item': None,  # 预测时不需要标签
            'neg_item': None
        }
        
        # 获取预测分数
        scores, _ = model(batch)  # [batch_size, n_items]
        
        # 只保留候选物品的分数
        if candidate_items is not None:
            scores = scores[:, candidate_items]
        
        # 根据分数排序
        _, indices = torch.topk(scores, 100)
        
        return indices + 1

# 使用示例：
def recommend_items(model, user_seq, top_k=10, candidate_items=None):
    """
    为用户推荐 top-k 个物品
    
    Args:
        model: 训练好的 RankBoost 模型
        user_seq: 用户的历史交互序列 [batch_size, seq_len]
        top_k: 推荐物品数量
        candidate_items: 候选物品列表（可选）
        
    Returns:
        recommended_items: 推荐的物品 ID 列表 [batch_size, top_k]
        scores: 对应的预测分数 [batch_size, top_k]
    """
    ranked_items = rank_items(model, user_seq, candidate_items)
    
    # 返回 top-k 推荐结果
    return ranked_items[:, :top_k]

# 实际使用示例
if __name__ == "__main__":
    # 假设我们已经有了训练好的模型
    model = RankBoost({
        'device': 'cpu',
        'hidden_dim': 128,
        'n_weak_learners': 10,
        'lr': 0.001
    }, 1000)  # 加载训练好的模型

    # 用户历史序列 - 现在支持批处理
    user_seqs = [[101, 202, 303, 404], [505, 606, 707, 808]]  # 示例序列

    # 1. 对所有物品进行排序
    ranked_all = rank_items(model, user_seqs)
    print("全部物品排序结果：")
    for batch_idx, batch_items in enumerate(ranked_all):
        print(f"\n用户 {batch_idx + 1} 的排序结果:")
        for item in batch_items[:5]:
            print(f"物品 ID: {item}")

    # 2. 只对部分候选物品进行排序
    candidate_items = [501, 602, 703, 804, 905]
    ranked_candidates = rank_items(model, user_seqs, candidate_items)
    print("\n候选物品排序结果：")
    for batch_idx, batch_items in enumerate(ranked_candidates):
        print(f"\n用户 {batch_idx + 1} 的候选排序结果:")
        for item in batch_items[:5]:
            print(f"物品 ID: {item}")

    # 3. 获取 top-k 推荐
    top_k = 5
    recommended_items = recommend_items(model, user_seqs, top_k)
    print(f"\nTop-{top_k} 推荐结果：")
    for batch_idx, batch_items in enumerate(recommended_items):
        print(f"\n用户 {batch_idx + 1} 的推荐结果:")
        for item in batch_items[:5]:
            print(f"物品 ID: {item}")
