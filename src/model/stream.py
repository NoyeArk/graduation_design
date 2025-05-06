import torch
import torch.nn as nn


class StackingModel(nn.Module):
    def __init__(self, args, data_args, n_user):
        super(StackingModel, self).__init__()
        self.args = args
        self.data_args = data_args
        self.n_user = n_user
        self.n_base_model = len(data_args['base_model'])
        self.device = torch.device(args['device'])

        self.base_model_projection = nn.Sequential(
            nn.Linear(self.n_base_model, 64),  # 第一层线性变换
            nn.ReLU(),                         # 激活函数
            nn.Dropout(0.5),                   # 添加dropout层
            nn.Linear(64, 32),                 # 第二层线性变换
            nn.ReLU(),                         # 激活函数
            nn.Dropout(0.5),                   # 添加dropout层
            nn.Linear(32, 1)                   # 输出层
        )
        self.to(self.device)

    def forward(self, batch, is_test=False):
        if is_test:
            all_item_scores = batch['all_item_scores'].permute(0, 2, 1)  # [bc, n_item, k]
            pred_scores = self.base_model_projection(all_item_scores)  # [bc, n_item, 1]
            return pred_scores.squeeze(-1)
        pos_scores = self.base_model_projection(batch['pos_label'].float())  # [bc, 1]
        neg_scores = self.base_model_projection(batch['neg_label'].float())  # [bc, 1]
        loss = -torch.sum(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        return loss
