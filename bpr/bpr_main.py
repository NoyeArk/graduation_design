import yaml
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from bpr.model.ensrec import SeqLearn
from data import Data, BPRLoss
from torch.utils.tensorboard import SummaryWriter


def train(config, model, train_loader, criterion, optimizer):
    # 冻结LLM参数
    for name, param in model.named_parameters():
        if name.startswith('cem.llm'):
            param.requires_grad = False

    # 计算可训练总参数量
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练总参数量: {total_trainable_params:,}")

    print("---------------------------")
    # 查看每层可训练参数
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"层: {name}, 形状: {param.shape}, 可训练参数量: {param.numel()}, {param.requires_grad}")

    train.writer = SummaryWriter('runs/bpr_train_test')
    train.global_step = 0

    for epoch in range(config['epoch']):
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epoch']}") as pbar:
            for batch in pbar:
                users, user_seq, pos_items, neg_items, pos_labels, neg_labels, base_model_preds = batch
                optimizer.zero_grad()
                pos_scores, neg_scores = model(users, user_seq, pos_items, neg_items, pos_labels, neg_labels, base_model_preds)
                loss = criterion(pos_scores, neg_scores)
                loss.backward()
                optimizer.step()

                total_grad = 0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        total_grad += grad_norm

                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    total_grad=f"{total_grad:.4f}",
                    pos_scores=f"{pos_scores.mean().item():.4f}",
                    neg_scores=f"{neg_scores.mean().item():.4f}"
                )

                train.writer.add_scalar('训练/损失', loss.item(), train.global_step)
                train.global_step += 1

        if (epoch + 1) % 3 == 0:
            torch.save(model.state_dict(), f"ckpt_sem/sem_epoch{epoch+1}.pth")
            print(f"模型已保存: ckpt_sem/sem_epoch{epoch+1}.pth")


if __name__ == '__main__':
    with open("config/sem.yaml", 'r', encoding='utf-8') as f:
        args = yaml.unsafe_load(f)
    print(args)

    data = Data(args['data'])
    train_loader = DataLoader(data.train_dataset, batch_size=args['batch_size'], shuffle=True)

    criterion = BPRLoss()
    model = SeqLearn(args['model'], args['data'], data.n_user, data.n_item)
    optimizer = torch.optim.Adam(model.parameters(), lr=args['model']['lr'])

    # model.load_state_dict(torch.load("ckpt_score_sum/bpr_epoch3.pth"))
    train(args, model, train_loader, criterion, optimizer)

    torch.save(model.state_dict(), "ckpt_sem/sem_final.pth")
