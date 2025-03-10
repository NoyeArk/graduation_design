import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from module.learn import ItemTower
from module.llm_cem import ContentExtractionModule

# 'ACF', 'FDSA', 'HARNN' 需要 attribute 信息，Caser', 'PFMC', 'SASRec' 仅依赖于序列信息。
print_train = False  # 是否输出 train 上的验证结果（过拟合解释）
base_models = ['acf', 'fdsa', 'harnn', 'caser', 'pfmc', 'sasrec', 'anam']


class SeqLearn(nn.Module):
    def __init__(self, args, data):
        super(SeqLearn, self).__init__()
        self.args = args
        self.data = data
        self.n_user = self.data.entity_num['user']
        self.n_item = self.data.entity_num['item']
        self.learning_rate = args['lr']
        self.hidden_dim = args['hidden_dim']
        self.reg_weight = args['lamda']
        self.optimizer_type = args['optimizer']
        self.n_base_model = len(base_models)
        self.seq_max_len = self.data.args['maxlen']
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
                                   max_length=self.data.args['maxlen'],
                                   data_filepath=f"{self.data.args['item_path']}",
                                   cache_path=f"{self.data.args['item_emb_path']}",
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

        for b in range(batch_size):
            for k in range(n_base_model):
                for s in range(seq_len):
                    item_id = base_focus[b, k, s].item()
                    # 因为物品ID需要+1才与movies.dat匹配
                    if 0 <= item_id < self.n_item:  # 确保ID在有效范围内
                        result[b, k, s] = self.item_tower.item_embeddings[item_id]

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
        user_id,
        input_seq,
        base_focus=None,
        is_train=True,
        positive_scores=None,
        negative_scores=None,
        all_items_scores=None
    ):
        """
        前向传播

        Args:
            user_id (`torch.Tensor`): 用户ID
            input_seq (`torch.Tensor`): 输入序列，形状为 [batch_size, seq_len]
            base_focus (`torch.Tensor`, optional): 基模型推理结果表示，形状为 [batch_size, n_base_model, seq_len]
            is_train (`bool`, optional): 是否训练
            positive_scores (`torch.Tensor`, optional): 正样本得分，形状为 [batch_size, n_base_model]
            negative_scores (`torch.Tensor`, optional): 负样本得分，形状为 [batch_size, n_base_model]
            all_items_scores (`torch.Tensor`, optional): 所有物品的得分，形状为 [batch_size, n_item]

        Returns:
            `dict`: 包含损失和预测结果的字典
        """
        # user 侧
        user_emb = self.user_embeddings(user_id)  # bc, hidden_dim  
        user_interaction = self.item_tower(input_seq)  # bc, seq_len, hidden_dim
        preference = self.dien_with_self_attention(user_interaction)[:, -1, :] + user_emb  # bc, hidden_dim

        # item 侧
        base_model_focus_llm = self._convert_focus_to_llm_embeddings(base_focus)
        each_model_emb = self.llm_projection(base_model_focus_llm)
        basemodel_emb = each_model_emb.mean(dim=2)

        # 计算基模型权重
        wgts_org = torch.sum(preference.unsqueeze(1) * basemodel_emb, dim=-1)
        wgts = F.softmax(wgts_org, dim=-1)

        results = {'wgts': wgts}

        if is_train and positive_scores is not None and negative_scores is not None:
            positive_scores = torch.sum(positive_scores * wgts, dim=1)
            negative_scores = torch.sum(negative_scores * wgts.unsqueeze(1), dim=-1)
            loss_rec = -torch.sum(torch.sigmoid(positive_scores - negative_scores))

            loss_reg = 0
            # for param in self.parameters():
            #     loss_reg += self.reg_weight * torch.sum(param ** 2)

            if self.args['div_module'] == 'AEM-cov':
                model_emb = basemodel_emb
                cov_idx = torch.ones(1, self.n_base_model, self.n_base_model, device=self.device)
                cov_idx = cov_idx - torch.eye(self.n_base_model, device=self.device).unsqueeze(0)
                cov_div1 = torch.square(
                    torch.sum(
                        model_emb.unsqueeze(1) * model_emb.unsqueeze(2),
                        dim=-1
                    )
                )
                l2 = torch.sum(model_emb ** 2, dim=-1)
                cov_div2 = torch.bmm(l2.unsqueeze(-1), l2.unsqueeze(1))
                cov = cov_div1 / cov_div2

            elif self.args['div_module'] == 'cov':
                model_emb = basemodel_emb
                cov_wgt = torch.cat([wgts.unsqueeze(1) + wgts.unsqueeze(2)], dim=0)
                cov_idx = torch.ones(1, self.n_base_model, self.n_base_model, device=self.device)
                cov_idx = cov_idx - torch.eye(self.n_base_model, device=self.device).unsqueeze(0)

                cov_div = torch.square(
                    torch.sum(
                        model_emb.unsqueeze(1) * model_emb.unsqueeze(2),
                        dim=-1
                    )
                )
                cov = cov_idx * (1 - cov_div)

            loss_diversity = -self.args['tradeoff'] * torch.sum(cov)
            loss = loss_rec # + loss_reg  + loss_diversity

            results.update({
                'loss': loss,
                'loss_rec': loss_rec,
                'loss_diversity': loss_diversity,
                'loss_reg': loss_reg
            })

        if all_items_scores is not None:
            # 计算所有物品的得分
            out = torch.sum(wgts.unsqueeze(2) * all_items_scores, dim=1)
            values, indices = torch.topk(out, 200)
            results.update({
                'pred_values': values,
                'pred_indices': indices
            })

        return results

    def partial_fit(self, data_dict):
        """
        拟合一个批次。

        Args:
            data_dict (`dict`): 数据字典

        Returns:
            tuple: (loss_rec, loss_diversity)
        """
        # 将数据转移到设备
        user_id = torch.tensor(data_dict['u'], dtype=torch.long, device=self.device)
        input_seq = torch.tensor(data_dict['seq'], dtype=torch.long, device=self.device)
        positive_scores = torch.tensor(data_dict['meta_pos'], dtype=torch.float, device=self.device)
        negative_scores = torch.tensor(data_dict['meta_neg'], dtype=torch.float, device=self.device)

        # 转换 base_focus 为大模型嵌入
        base_focus = torch.tensor(data_dict['base_focus'], dtype=torch.long, device=self.device)

        self.train()
        self.optimizer.zero_grad()
        results = self.forward(
            user_id=user_id,
            input_seq=input_seq,
            base_focus=base_focus,
            is_train=True,
            positive_scores=positive_scores,
            negative_scores=negative_scores
        )

        results['loss'].backward()
        
        # 打印所有参数的梯度统计信息
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"{name}:")
                print(f"  梯度是否存在: {param.grad is not None}")
                if param.grad is not None:
                    print(f"  梯度形状: {param.grad.shape}")
                    print(f"  梯度均值: {param.grad.mean().item()}")
                    print(f"  梯度标准差: {param.grad.std().item()}")
                    print(f"  梯度最小值: {param.grad.min().item()}")
                    print(f"  梯度最大值: {param.grad.max().item()}")
                    print(f"  梯度范数: {param.grad.norm().item()}")
                    print(f"  是否有NaN: {torch.isnan(param.grad).any().item()}")
                    print(f"  是否有Inf: {torch.isinf(param.grad).any().item()}")

        self.optimizer.step()

        return results['loss_rec'].item(), results['loss_diversity'].item(), 0

    def topk(self, user_item_pairs, last_interaction, items_score, base_focus):
        """
        计算 topk 得分。

        Args:
            user_item_pairs (np.ndarray): 用户-物品对
            last_interaction (np.ndarray): 最后一次交互
            items_score (np.ndarray): 物品得分
            base_focus (np.ndarray): 基模型表示

        Returns:
            tuple: (pred_item, wgts)
        """
        user_id = torch.tensor(user_item_pairs[:, 0], dtype=torch.long, device=self.device)
        input_seq = torch.tensor(last_interaction, dtype=torch.long, device=self.device)
        meta_all_items = torch.tensor(items_score, dtype=torch.float, device=self.device)

        base_focus = torch.tensor(base_focus, dtype=torch.long, device=self.device)

        self.eval()
        with torch.no_grad():
            results = self.forward(
                user_id=user_id,
                input_seq=input_seq,
                base_focus=base_focus,
                is_train=False,
                all_items_scores=meta_all_items
            )

        return results['pred_indices'].cpu().numpy(), results['wgts'].cpu().numpy()

    def save_model(self, save_path):
        torch.save(self.state_dict(), save_path)
        print(f"模型已保存到: {save_path}")

    def load_model(self, load_path):
        self.load_state_dict(torch.load(load_path))
        print(f"模型已从以下位置加载: {load_path}")
