import torch
import torch.nn as nn
import torch.nn.functional as F


base_models = ['acf', 'fdsa', 'harnn', 'caser', 'pfmc', 'sasrec', 'anam']


class Sem(nn.Module):
    def __init__(self, args, data_args, n_user, n_item):
        super(Sem, self).__init__()
        self.args = args
        self.data_args = data_args
        self.n_user = n_user
        self.n_item = n_item
        self.hidden_dim = args['hidden_dim']
        self.learning_rate = args['lr']
        self.reg_weight = args['lamda']
        self.optimizer_type = args['optimizer']
        self.n_base_model = len(base_models)
        self.seq_max_len = data_args['maxlen']
        self.device = args['device']

        # 初始化嵌入层
        self.user_embeddings = nn.Embedding(self.n_user, self.hidden_dim)
        self.item_embeddings = nn.Embedding((self.n_item + 1), self.hidden_dim)
        nn.init.normal_(self.user_embeddings.weight, 0, 0.01)
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)

        self.base_model_embeddings = nn.Parameter(torch.randn(1, self.n_base_model, self.hidden_dim) * 0.01)
        self.seq_weights = nn.Parameter(torch.randn(1, 1, self.seq_max_len, 1) * 0.01)

        # 用户序列建模
        self.user_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=2,
                dim_feedforward=self.hidden_dim,
                dropout=0.5
            ),
            num_layers=2
        )
        self.pos_embedding = nn.Embedding(self.seq_max_len, self.hidden_dim)

        self.to(self.device)
        print(f"model.device: {self.device}")

    def forward(self, user_ids, user_seq, pos_items, neg_items, pos_scores, neg_scores, base_model_preds):
        self.item_embeddings = self.item_embeddings.to(self.device)
        seq_emb = self.item_embeddings(user_seq)
        user_emb = self.user_embeddings(user_ids)  # [batch_size, hidden_dim]

        # 添加位置编码
        positions = torch.arange(self.seq_max_len, device=self.device).expand(user_seq.size(0), -1)
        seq_emb = seq_emb + self.pos_embedding(positions)

        # 创建注意力掩码
        mask = (user_seq == -1)
        output = self.user_encoder(seq_emb.transpose(0,1), src_key_padding_mask=mask).transpose(0,1)
        preference = output[:,-1,:] + user_emb

        base_model_emb = self.item_embeddings(base_model_preds)  # [batch_size, n_base_model, seq_len, hidden_dim]

        # 时间衰减权重
        time_weights = 1.0 / torch.log2(torch.arange(self.seq_max_len, device=self.device) + 2)
        time_weights = time_weights.view(1, 1, -1, 1)

        basemodel_emb = self.base_model_embeddings + torch.sum(time_weights * base_model_emb, dim=2)

        # 计算基模型权重
        wgts_org = torch.sum(preference.unsqueeze(1) * basemodel_emb, dim=-1)  # [batch_size, n_base_model]
        wgts = F.softmax(wgts_org, dim=-1)

        # 计算正负样本得分
        pos_pred = torch.sum(pos_scores * wgts, dim=1)  # [batch_size]
        neg_pred = torch.sum(neg_scores * wgts.unsqueeze(1), dim=-1)  # [batch_size, num_neg]

        # 计算正负样本损失
        loss_rec = -torch.sum(torch.sigmoid((pos_pred - neg_pred)))

        # 计算正则化损失
        loss_reg = 0
        for param in self.parameters():
            if param.requires_grad:
                loss_reg += self.args['lamda'] * torch.nn.functional.mse_loss(param, torch.zeros_like(param))

        # 计算多样性损失
        model_emb = basemodel_emb  # [batch_size, n_base_model, hidden_dim]
        cov_wgt = torch.detach(wgts.unsqueeze(1) + wgts.unsqueeze(2))  # [batch_size, n_base_model, n_base_model]
        cov_idx = (1 - torch.eye(self.n_base_model, device=self.device)).unsqueeze(0)  # [1, n_base_model, n_base_model]
        model_emb_1 = model_emb.unsqueeze(1)  # [batch_size, 1, n_base_model, hidden_dim]
        model_emb_2 = model_emb.unsqueeze(2)  # [batch_size, n_base_model, 1, hidden_dim]
        cov_div = torch.sum(model_emb_1 * model_emb_2, dim=-1).pow(2)  # [batch_size, n_base_model, n_base_model]

        # 计算最终的多样性损失
        self.cov = cov_idx * (1 - cov_div)  # [batch_size, n_base_model, n_base_model]
        self.cov = self.cov * cov_wgt
        loss_diversity = -self.args['tradeoff'] * torch.sum(self.cov)

        loss = loss_rec + loss_reg + loss_diversity

        return pos_pred, neg_pred, wgts, loss

    def predict(self, user_ids, user_seq, all_item_scores, base_model_preds):
        """
        预测所有物品得分
        """
        pos_pred, neg_pred, wgts, loss = self.forward(user_ids, user_seq, None, None, all_item_scores, None, base_model_preds)
        pred_scores = torch.sum(wgts.unsqueeze(2) * all_item_scores, dim=1)  # [batch_size, n_item]
        return pred_scores

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
