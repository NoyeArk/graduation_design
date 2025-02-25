import torch
import torch.nn as nn
import torch.nn.functional as F


class NextKItemPredictor(nn.Module):
    """
    用于预测用户接下来k个物品的推荐模型
    """
    def __init__(self, num_users, num_items, embedding_dim, num_next_items):
        """
        初始化模型参数

        Args:
            num_users (`int`): 用户数量
            num_items (`int`): 物品数量  
            embedding_dim (`int`): 嵌入维度
            num_next_items (`int`): 预测接下来的物品数量
        """
        super(NextKItemPredictor, self).__init__()

        self.user_embeddings = torch.nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = torch.nn.Embedding(num_items, embedding_dim)

        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim * 3, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

        self.num_next_items = num_next_items

    def forward(self, user_id, item_history, item_id):
        """
        前向传播
        
        Args:
            user_id (`torch.Tensor`): 用户ID, shape=[1]
            item_history (`torch.Tensor`): 用户历史交互物品序列, shape=[seq_len]
            item_id (`torch.Tensor`): 目标物品ID, shape=[1]

        Returns:
            推荐的下一个物品的概率分布, shape=[batch_size, num_items]
        """
        user_embed = self.user_embeddings(user_id)  # [embedding_dim]
        item_embed = self.item_embeddings(item_id)  # [embedding_dim]
        history_embed = self.item_embeddings(item_history).mean(dim=0, keepdim=True)  # [embedding_dim]

        concat_embed = torch.cat([user_embed, item_embed, history_embed], dim=1)  # [1, embedding_dim*3]
        score = self.predictor(concat_embed).unsqueeze(0)  # [1]

        return torch.sigmoid(score)

    def recommend_next_k_items(self, user_ids, item_history):
        """
        为用户推荐接下来k个物品
        
        Args:
            user_ids (`torch.Tensor`): 用户ID
            item_history (`torch.Tensor`): 用户历史交互物品序列
            
        Returns:
            推荐的k个物品ID, shape=[batch_size, k]
        """
        scores = self.forward(user_ids, item_history)
        _, indices = torch.topk(scores, self.num_next_items)

        return indices
    
    def save_checkpoint(self, path):
        """
        保存模型权重和优化器状态
        
        Args:
            path (`str`): 保存路径
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """
        加载模型权重和优化器状态
        
        Args:
            path (`str`): 加载路径
        """
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        if hasattr(self, 'optimizer') and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class BPRWithHistory(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(BPRWithHistory, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

    def forward(self, user_id, pos_item_id, neg_item_id, item_history):
        user_embed = self.user_embeddings(user_id)  # [embedding_dim]
        pos_item_embed = self.item_embeddings(pos_item_id)  # [embedding_dim]
        neg_item_embed = self.item_embeddings(neg_item_id)  # [embedding_dim]
        
        # History embedding
        history_embed = self.item_embeddings(item_history).mean(dim=0)  # [embedding_dim]

        # Combine embeddings
        user_feature = torch.cat([user_embed, history_embed], dim=0)  # [embedding_dim*2]

        # Calculate scores
        pos_score = torch.dot(user_feature, pos_item_embed)
        neg_score = torch.dot(user_feature, neg_item_embed)

        return pos_score, neg_score

    def bpr_loss(self, pos_score, neg_score):
        return -torch.log(torch.sigmoid(pos_score - neg_score)).mean()
