import torch
import torch.nn as nn
import torch.nn.functional as F

from module.dien import DIEN


class EnsRec(nn.Module):
    def __init__(self, args, data_args, n_user, n_item):
        super(EnsRec, self).__init__()
        self.args = args
        self.is_toys_data = data_args['name'] == 'Toys_and_Games'
        self.n_user = n_user
        self.n_item = n_item + 1
        self.learning_rate = args['lr']
        self.hidden_dim = args['hidden_dim']
        self.n_base_model = len(data_args['base_model'])
        self.seq_max_len = data_args['maxlen']
        self.device = torch.device(args['device'])

        self.dien = DIEN(args['hidden_dim'])
        self.user_embeddings = nn.Embedding(self.n_user, self.hidden_dim)
        self.item_embeddings = nn.Embedding(self.n_item, self.hidden_dim)
        self.base_model_emb = nn.Parameter(torch.randn(1, self.n_base_model, self.hidden_dim))

        self.to(self.device)

    def forward(self, batch, is_test=False):
        """
        前向传播

        Args:
            batch (`dict`): 一个字典，包含以下键值
            - user_id (`torch.Tensor`): 用户ID，形状为 [batch_size]
            - user_seq (`torch.Tensor`): 用户序列，形状为 [batch_size, seq_len]
            - pos_item (`torch.Tensor`): 物品ID，形状为 [batch_size]
            - neg_item (`torch.Tensor`): 负样本物品ID，形状为 [batch_size]
            - pos_label (`torch.Tensor`): 所有物品的得分，形状为 [batch_size]
            - neg_label (`torch.Tensor`): 所有物品的得分，形状为 [batch_size]
            - base_model_preds (`torch.Tensor`): 基模型预测，形状为 [batch_size, k, 100]

        Returns:
            `dict`: 包含损失和预测结果的字典
        """
        user_emb = self.user_embeddings(batch['user_id'])  # bc, dim
        preference = user_emb  # bc, dim

        # 计算基模型权重
        # [bc, n_base_model, dim] @ [bc, 1, dim] -> [bc, n_base_model, 1]
        preference = preference.unsqueeze(1).transpose(-2, -1)  # [bc, dim, 1]
        wgts_org = torch.matmul(self.base_model_emb, preference).squeeze(-1)  # [bc, n_base_model]
        wgts = F.softmax(wgts_org, dim=-1)  # [bc, n_base_model]

        if is_test:
            pred_all_item_scores = torch.matmul(wgts.unsqueeze(1), batch['all_item_scores']).squeeze(1)
            return pred_all_item_scores

        pos_scores = torch.sum(batch['pos_label'] * wgts, dim=1)  # bc
        neg_scores = torch.sum(batch['neg_label'] * wgts, dim=1)  # bc
        loss = -torch.sum(torch.log(torch.sigmoid(pos_scores - neg_scores)))

        if self.is_toys_data:
            loss += torch.sum((wgts - 1).pow(2))  # 惩罚专家失衡

        if self.args['use_div']:
            cov_wgt = torch.detach(wgts.unsqueeze(1) + wgts.unsqueeze(2))  # [bc, n_base_model, n_base_model]
            cov_mask = (1 - torch.eye(self.n_base_model, device=self.device)).unsqueeze(0)  # [1, n_base_model, n_base_model]
            model_emb1 = self.base_model_emb.unsqueeze(1)  # [bc, 1, n_base_model, dim]
            model_emb2 = self.base_model_emb.unsqueeze(2)  # [bc, n_base_model, 1, dim]
            cov_div = torch.sum(model_emb1 * model_emb2, dim=-1).pow(2)  # [bc, n_base_model, n_base_model]

            loss_div = cov_mask * cov_div * cov_wgt # [bc, n_base_model, n_base_model]
            loss_div = self.args['div_tradeoff'] * torch.sum(loss_div)
            loss += loss_div

        if self.args['use_reg']:
            loss_reg = 0
            for param in self.parameters():
                if param.requires_grad:
                    loss_reg += self.args['reg_tradeoff'] * F.mse_loss(param, torch.zeros_like(param))
            loss += loss_reg

        return loss
