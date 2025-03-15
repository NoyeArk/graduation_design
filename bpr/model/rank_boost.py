import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict


class RankBoost(nn.Module):
    def __init__(self, args):
        super(RankBoost, self).__init__()
        self.device = args.device
        self.n_items = args.n_items
        self.dim = args.hidden_dim
        self.n_weak_learners = args.n_weak_learners  # 弱学习器数量
        self.learning_rate = args.lr
        
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
        
        # 项目嵌入层
        self.item_embeddings = nn.Embedding(self.n_items + 1, self.dim, padding_idx=0)
        
    def forward(self, users, user_seq, pos_items, neg_items, pos_labels, neg_labels, base_model_preds):
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
            
        # 计算损失
        loss = self.rankboost_loss(final_scores, pos_items, neg_items)
        
        return final_scores, loss
    
    def rankboost_loss(self, scores, pos_items, neg_items):
        """
        计算 RankBoost 损失函数
        """
        batch_size = scores.size(0)
        
        # 获取正样本和负样本的得分
        pos_scores = torch.gather(scores, 1, pos_items.unsqueeze(1))  # [batch_size, 1]
        neg_scores = torch.gather(scores, 1, neg_items)  # [batch_size, num_neg]
        
        # 计算排序损失
        diff = pos_scores - neg_scores  # [batch_size, num_neg]
        loss = torch.mean(torch.exp(-diff))  # 指数损失函数
        
        return loss
    
    def update_alphas(self, scores, pos_items, neg_items, weights):
        """
        更新弱分类器的组合系数
        """
        with torch.no_grad():
            # 计算每个弱分类器的错误率
            errors = []
            for scores in scores:
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
