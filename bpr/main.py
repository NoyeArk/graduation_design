import os
import sys
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import Data, SeqBPRDataset
from model.model_factory import get_model


def nDCG(rec_items, test_set):
    DCG = lambda x: np.sum(x / np.log(np.arange(2, len(x) + 2)))
    def get_implict_matrix(rec_items, test_set):
        rel_matrix = [[0] * rec_items.shape[1] for _ in range(rec_items.shape[0])]
        for user in range(len(test_set)):
            for index, item in enumerate(rec_items[user]):
                if item in test_set[user]:
                    rel_matrix[user][index] = 1
        return np.array(rel_matrix)
    rel_matrix = get_implict_matrix(rec_items, test_set)
    ndcgs = []
    for user in range(len(test_set)):
        rels = rel_matrix[user]
        dcg = DCG(rels)
        idcg = DCG(sorted(rels, reverse=True))
        ndcg = dcg / idcg if idcg != 0 else 0
        ndcgs.append(ndcg)
    return ndcgs


def train(args, data, model, train_loader, test_loader, optimizer):
    for name, param in model.named_parameters():
        if name.startswith('cem.llm'):
            param.requires_grad = False

    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练总参数量: {total_trainable_params:,}")

    train.writer = SummaryWriter(f'runs/{args["model"]["name"]}_train')
    train.global_step = 0

    for epoch in range(args['epoch']):
        model.train()
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args['epoch']}") as pbar:
            for batch in pbar:
                optimizer.zero_grad()
                loss = model(batch)
                loss.backward()
                optimizer.step()

                total_grad = 0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        total_grad += param.grad.norm().item()

                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    total_grad=f"{total_grad:.4f}"
                )

                train.writer.add_scalar('训练/损失', loss.item(), train.global_step)
                train.global_step += 1

        ndcg = test(data, model, test_loader, args['topk'])
        print(f"测试集/nDCG: {ndcg:.4f}")

        if not os.path.exists(f"ckpt_{args['model']['name']}"):
            os.makedirs(f"ckpt_{args['model']['name']}")
        ckpt_name = f"ckpt_{args['model']['name']}/epoch{epoch+11}_{round(ndcg, 4)}.pth"
        torch.save(model.state_dict(), ckpt_name)
        print(f"模型已保存: {ckpt_name}")


def test(data, model, test_loader, topk):
    model.eval()
    ndcg_scores = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="计算测试集指标"):
            all_scores = model(batch, is_test=True)
            _, indices = torch.topk(all_scores, topk)

            for i in range(len(batch['user_id'])):
                user_id = batch['user_id'][i].item()
                pos_item = batch['pos_item'][i].item()

                true_item_ids = data.user_interacted_item_ids[user_id]
                true_item_ids = true_item_ids[true_item_ids.index(data.item_to_id[pos_item]) + 1:]
                # true_item_ids = true_item_ids[true_item_ids.index(pos_item) + 1:]

                predicted_item_ids = np.array([indices[i].cpu().numpy().tolist()])
                ndcg = nDCG(np.array(predicted_item_ids), [true_item_ids])

                ndcg_scores.append(ndcg)
    return np.mean(ndcg_scores)


if __name__ == '__main__':
    with open(f"config/{sys.argv[1]}", 'r', encoding='utf-8') as f:
        args = yaml.unsafe_load(f)
    print(args)

    data = Data(args['data'])
    train_samples = np.load(f'datasets/{args["data"]["name"]}/train_samples.npy', allow_pickle=True)
    test_samples = np.load(f'datasets/{args["data"]["name"]}/test_samples.npy', allow_pickle=True)
    train_dataset = SeqBPRDataset(train_samples, args['data']['device'])
    test_dataset = SeqBPRDataset(test_samples, args['data']['device'], is_test=True)
    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False)

    model = get_model(args['model']['type'], args['model'], args['data'], data.n_user, 3952, data.id_to_item)
    optimizer = torch.optim.Adam(model.parameters(), lr=args['model']['lr'])

    model.load_state_dict(torch.load("ckpt_ensrec_kuairec/epoch10_0.4597.pth"))
    train(args, data, model, train_loader, test_loader, optimizer)
