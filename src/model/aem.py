import torch
import torch.nn as nn
import torch.nn.functional as F


class AEM(nn.Module):
    def __init__(self, args, data_args, n_user, n_item):
        super(AEM, self).__init__()
        self.args = args
        self.data_args = data_args
        self.n_user = n_user
        self.n_item = n_item + 1
        self.hidden_dim = args['hidden_dim']
        self.n_base_model = len(data_args['base_model'])
        self.seq_max_len = self.data_args['maxlen']
        self.device = torch.device(args['device'])

        self.user_embeddings = nn.Embedding(self.n_user, self.hidden_dim)
        self.item_embeddings = nn.Embedding(self.n_item, self.hidden_dim)
        nn.init.normal_(self.user_embeddings.weight, 0, 0.01)
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)

        self.classifier_embeddings = nn.Parameter(torch.randn(self.n_base_model, self.hidden_dim))

        self.to(torch.device(args['device']))

    def forward(self, batch, is_test=False):
        item_emb = self.item_embeddings(batch['pos_item'])  # [bc, dim]
        wgts = F.softmax(torch.matmul(item_emb, self.classifier_embeddings.T), dim=1)  # [bc, n_base_model]

        if is_test:
            pred_all_item_scores = torch.matmul(wgts.unsqueeze(1), batch['all_item_scores']).squeeze(1)
            return pred_all_item_scores

        # 计算正负样本得分
        pos_scores = torch.sum(batch['pos_label'] * wgts, dim=1)  # bc
        neg_scores = torch.sum(batch['neg_label'] * wgts, dim=1)  # bc

        loss = -torch.sum(torch.log(torch.sigmoid(pos_scores - neg_scores)))

        return loss
