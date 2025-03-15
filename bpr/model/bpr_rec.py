import torch
import torch.nn as nn
import torch.nn.functional as F

from module.learn import ItemTower
from module.llm_cem import ContentExtractionModule

base_models = ['acf', 'fdsa', 'harnn', 'caser', 'pfmc', 'sasrec', 'anam']


class BPRSeqLearn(nn.Module):
    def __init__(self, args, data_args, n_user, n_item):
        super(BPRSeqLearn, self).__init__()
        self.args = args
        self.data_args = data_args
        self.n_user = n_user
        self.n_item = n_item
        self.learning_rate = args['lr']
        self.hidden_dim = args['hidden_dim']
        self.reg_weight = args['lamda']
        self.optimizer_type = args['optimizer']
        self.n_base_model = len(base_models)
        self.seq_max_len = self.data_args['maxlen']
        self.device = torch.device("cuda:0")
        self.cem = ContentExtractionModule()
        self._initialize_weights()
        self.to(self.device)
        self._initialize_optimizer()

    def _initialize_weights(self):
        self.user_embeddings = nn.Embedding(self.n_user, self.hidden_dim)
        self.seq_weights = nn.Parameter(torch.randn(1, 1, self.seq_max_len, 1) * 0.01)

        # 添加ItemTower用于获取用户嵌入
        self.item_tower = ItemTower(hidden_factor=self.hidden_dim,
                                   pretrained_model_name=self.args['pretrain_llm'],
                                   max_length=self.data_args['maxlen'],
                                   data_filepath=f"{self.data_args['item_path']}",
                                   cache_path=f"{self.data_args['item_emb_path']}",
                                   device=self.device)

        # LLM投影层
        self.llm_projection = nn.Linear(self.item_tower.item_embeddings.shape[-1], self.hidden_dim)

        # 初始化权重
        nn.init.normal_(self.user_embeddings.weight, 0, 0.01)
        # DIEN + attention
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.attention_layer = nn.Linear(self.hidden_dim, 1)
        self.self_attention_q = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.self_attention_k = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.self_attention_v = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.self_attention_output = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.layer_norm1 = nn.LayerNorm(self.hidden_dim)
        self.layer_norm2 = nn.LayerNorm(self.hidden_dim)
        self.augru = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)

        self.trans_layer = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        self.out_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )

    def _initialize_optimizer(self):
        if self.optimizer_type == 'AdamOptimizer':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == 'AdagradOptimizer':
            self.optimizer = torch.optim.Adagrad(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)

    def _convert_focus_to_llm_embeddings(self, base_focus):
        """
        将base_focus中的物品ID转换为大模型嵌入

        Args:
            base_focus (`torch.Tensor`): 形状为[bc, n_base_model, seq_len]的基模型物品ID

        Returns:
            `torch.Tensor`: 形状为[bc, n_base_model, seq_len, hidden_size]的大模型嵌入
        """
        batch_size, n_base_model, seq_len = base_focus.shape
        hidden_size = self.item_tower.item_embeddings.shape[-1]

        # 初始化结果张量
        result = torch.zeros((batch_size, n_base_model, seq_len, hidden_size), device=self.device)

        # 将base_focus展平为2D张量以便一次性索引
        flat_focus = base_focus.reshape(-1)
        
        # 创建掩码标识有效的item_id
        valid_mask = (flat_focus >= 0) & (flat_focus < self.n_item)
        valid_indices = flat_focus[valid_mask]
        
        # 获取对应的嵌入
        valid_embeddings = self.item_tower.item_embeddings[valid_indices]
        
        # 将结果放回原始形状
        result_flat = result.reshape(-1, hidden_size)
        result_flat[valid_mask] = valid_embeddings
        
        # 恢复原始形状
        result = result_flat.reshape(batch_size, n_base_model, seq_len, hidden_size)

        return result

    def dien_with_self_attention(self, input_seq):
        """
        计算增强版DIEN(Deep Interest Evolution Network)的输出，增加了自注意力机制。
        
        Args:
            input_seq (`torch.Tensor`): 输入序列 [batch_size, seq_len, hidden_dim]

        Returns:
            `torch.Tensor`: 增强版DIEN的输出 [batch_size, seq_len, hidden_dim]
        """
        # GRU层提取兴趣
        gru_outputs, _ = self.gru(input_seq)

        # 创建注意力掩码
        batch_size, seq_len, _ = input_seq.shape
        valid_seq = torch.ones((batch_size, seq_len), device=input_seq.device)
        mask_a = valid_seq.unsqueeze(2)
        mask_b = valid_seq.unsqueeze(1)
        attention_mask = torch.bmm(mask_a, mask_b)

        # 多头自注意力
        q = self.self_attention_q(gru_outputs)
        k = self.self_attention_k(gru_outputs)
        v = self.self_attention_v(gru_outputs)

        # 计算注意力权重
        scores = torch.bmm(q, k.transpose(1, 2)) / (self.hidden_dim ** 0.5)
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)

        # 应用注意力
        self_attention_output = torch.bmm(attention_weights, v)
        self_attention_output = self.self_attention_output(self_attention_output)

        # 残差连接和层归一化
        self_attention_output = self_attention_output + gru_outputs
        self_attention_output = self.layer_norm1(self_attention_output)

        # 兴趣演化层
        att_weights = self.attention_layer(self_attention_output)
        att_weights = F.softmax(att_weights, dim=1)

        # 加权后的序列表示
        weighted_seq = self_attention_output * att_weights

        # 兴趣演化GRU
        final_outputs, _ = self.augru(weighted_seq)

        final_outputs = F.dropout(final_outputs, p=0.5, training=self.training)
        final_outputs = self.layer_norm2(final_outputs)

        return final_outputs

    def FFN(self, input_seq):
        """
        计算前馈神经网络的输出。

        Args:
            input_seq (`torch.Tensor`): 输入序列

        Returns:
            `torch.Tensor`: 前馈神经网络的输出
        """
        # 创建掩码
        mask = (input_seq != -1).float().unsqueeze(-1)

        # 处理输入序列
        input_seq_masked = torch.where(
            input_seq == -1,
            torch.zeros_like(input_seq) + self.n_item,
            input_seq
        )

        # 获取序列的嵌入表示
        seq_emb = self.sasrec_item_embeddings(input_seq_masked)

        # 位置编码
        positions = torch.arange(input_seq.size(1), device=self.device).unsqueeze(0).expand(input_seq.size(0), -1)
        pos_emb = self.pos_embeddings(positions)
        seq_emb = seq_emb + pos_emb

        # Dropout
        seq_emb = F.dropout(seq_emb, p=0.5, training=self.training)
        seq_emb = seq_emb * mask

        # 自注意力层
        for i in range(2):
            # 自注意力
            norm_seq = self.layer_norms[i*2](seq_emb)
            # 转置以适应PyTorch的多头注意力
            norm_seq_t = norm_seq.transpose(0, 1)
            attn_output, _ = self.attention_layers[i](
                norm_seq_t, norm_seq_t, norm_seq_t, 
                key_padding_mask=(input_seq == -1)
            )
            attn_output = attn_output.transpose(0, 1)
            seq_emb = seq_emb + attn_output

            # 前馈网络
            norm_seq = self.layer_norms[i*2+1](seq_emb)
            ff_output = self.feed_forward(norm_seq)
            seq_emb = seq_emb + ff_output
            seq_emb = seq_emb * mask

        seq_emb = self.layer_norms[-1](seq_emb)
        return seq_emb

    def forward(
        self,
        users,
        user_seq,
        pos_items,
        neg_items,
        pos_labels,
        neg_labels,
        base_model_preds
    ):
        """
        前向传播

        Args:
            users (`torch.Tensor`): 用户ID，形状为 [batch_size]
            user_seq (`torch.Tensor`): 用户序列，形状为 [batch_size, seq_len]
            pos_items (`torch.Tensor`): 物品ID，形状为 [batch_size]
            neg_items (`torch.Tensor`): 负样本物品ID，形状为 [batch_size]
            pos_items (`torch.Tensor`): 所有物品的得分，形状为 [batch_size]
            neg_items (`torch.Tensor`): 所有物品的得分，形状为 [batch_size]
            base_model_preds (`torch.Tensor`): 基模型预测，形状为 [batch_size, k, 100]

        Returns:
            `dict`: 包含损失和预测结果的字典
        """
        # 获取用户嵌入
        user_emb = self.user_embeddings(users)  # batch_size, hidden_dim

        # user 侧
        user_interaction = self.item_tower(user_seq)  # bc, seq_len, hidden_dim
        preference = self.dien_with_self_attention(user_interaction)[:, -1, :] + user_emb  # bc, hidden_dim

        # item 侧
        base_model_focus_llm = self._convert_focus_to_llm_embeddings(base_model_preds)  # bc, n_base_model, seq_len, hidden_dim
        each_model_emb = self.llm_projection(base_model_focus_llm)  # bc, n_base_model, seq_len, hidden_dim
        basemodel_emb = each_model_emb.mean(dim=2)  # bc, n_base_model, hidden_dim

        # 计算基模型权重
        wgts_org = torch.sum(preference.unsqueeze(1) * basemodel_emb, dim=-1)
        wgts = F.softmax(wgts_org, dim=-1)

        # 计算正负样本得分
        pos_scores = torch.sum(pos_labels * wgts, dim=1)  # bc
        neg_scores = torch.sum(neg_labels * wgts, dim=1)  # bc

        return pos_scores, neg_scores, self.loss(pos_scores, neg_scores)

    def predict(self, users, user_seq, pos_item, neg_item, all_item_scores, base_model_preds):
        """
        预测用户对物品的评分

        Args:
            users (`torch.Tensor`): 用户ID，形状为 [batch_size]
            user_seq (`torch.Tensor`): 用户序列，形状为 [batch_size, seq_len]
            pos_items (`torch.Tensor`): 物品ID，形状为 [batch_size]
            neg_items (`torch.Tensor`): 负样本物品ID，形状为 [batch_size]
            all_items_scores (`torch.Tensor`): 所有物品的得分，形状为 [batch_size, n_base_model, num_items]
            base_model_preds (`torch.Tensor`): 基模型预测，形状为 [batch_size, n_base_model, seq_len]

        Returns:
            `torch.Tensor`: 预测分数，形状为 [batch_size, num_items]
        """
        # 获取用户嵌入
        user_emb = self.user_embeddings(users)  # batch_size, hidden_dim

        # user 侧
        user_interaction = self.item_tower(user_seq)  # bc, seq_len, hidden_dim
        preference = self.dien_with_self_attention(user_interaction)[:, -1, :] + user_emb  # bc, hidden_dim

        # item 侧
        base_model_focus_llm = self._convert_focus_to_llm_embeddings(base_model_preds)  # bc, n_base_model, seq_len, hidden_dim
        each_model_emb = self.llm_projection(base_model_focus_llm)  # bc, n_base_model, seq_len, hidden_dim
        basemodel_emb = each_model_emb.mean(dim=2)  # bc, n_base_model, hidden_dim

        # 计算基模型权重
        wgts_org = torch.sum(preference.unsqueeze(1) * basemodel_emb, dim=-1)  # bc, n_base_model
        wgts = F.softmax(wgts_org, dim=-1)  # bc, n_base_model

        # 计算所有物品得分
        pred_all_item_scores = torch.sum(wgts.unsqueeze(2) * all_item_scores, dim=1)  # bc

        return pred_all_item_scores
    
    def loss(self, pos_scores, neg_scores):
        return -torch.sum(torch.log(torch.sigmoid(pos_scores - neg_scores)))

    def save_model(self, save_path):
        torch.save(self.state_dict(), save_path)
        print(f"模型已保存到: {save_path}")

    def load_model(self, load_path):
        self.load_state_dict(torch.load(load_path))
        print(f"模型已从以下位置加载: {load_path}")
