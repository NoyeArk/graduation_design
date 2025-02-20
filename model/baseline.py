import torch
import torch.nn as nn


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
            torch.nn.Linear(embedding_dim * 2, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_items)
        )

        self.num_next_items = num_next_items

    def forward(self, user_ids, item_history):
        """
        前向传播
        
        Args:
            user_ids (`torch.Tensor`): 用户ID, shape=[batch_size]
            item_history (`torch.Tensor`): 用户历史交互物品序列, shape=[batch_size, seq_len]
            
        Returns:
            推荐的下一个物品的概率分布, shape=[batch_size, num_items]
        """
        user_embed = self.user_embeddings(user_ids)  # [batch_size, embedding_dim]

        item_embeds = self.item_embeddings(item_history)  # [batch_size, seq_len, embedding_dim]
        item_embed_mean = torch.mean(item_embeds, dim=1)  # [batch_size, embedding_dim]

        concat_embed = torch.cat([user_embed, item_embed_mean], dim=1)  # [batch_size, embedding_dim*2]

        scores = self.predictor(concat_embed)  # [batch_size, num_items]

        return torch.sigmoid(scores)

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
