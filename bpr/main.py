import yaml
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import SeqLearn
from torch.utils.tensorboard import SummaryWriter
from data import BPRSampleGenerator, SeqBPRDataset, BPRLoss


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

    train.writer = SummaryWriter('runs/bpr_training')
    train.global_step = 0

    for epoch in range(config['epoch']):
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epoch']}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                users, user_seq, pos_items, neg_items, base_model_preds = batch
                optimizer.zero_grad()
                pos_scores, neg_scores = model(users, user_seq, pos_items, neg_items, base_model_preds)
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

                # 记录各项指标
                train.writer.add_scalar('训练/损失', loss.item(), train.global_step)
                
                # 更新全局步数
                train.global_step += 1

                # 每训练5个batch保存一次模型
                if (batch_idx + 1) % 5 == 0:
                    torch.save(model.state_dict(), f"ckpt/bpr_epoch{epoch+1}_batch{batch_idx+1}.pth")
                    print(f"模型已保存: ckpt/bpr_epoch{epoch+1}_batch{batch_idx+1}.pth")

                    # avg_ndcg = test(config, model, test_loader)
                    # print(f"测试集上的平均NDCG@{config['topk']}: {avg_ndcg:.4f}")

                    # train.writer.add_scalar(f'测试/NDCG@{config["topk"]}', avg_ndcg, train.global_step)


def test(config, model, test_loader):
    model.eval()
    # 计算每个用户的NDCG@k
    with torch.no_grad():
        ndcg_scores = []
        for batch in tqdm(test_loader, desc="计算测试集NDCG"):
            users, user_seq, pos_items, neg_items, base_model_preds = batch

            # 获取所有物品的预测分数
            all_items = torch.arange(len(generator.item_to_id)).to(config['model']['device'])
            all_scores = model.predict(users, user_seq, all_items, base_model_preds)

            _, indices = torch.topk(all_scores, config['topk'])

            for user_idx in range(len(users)):
                # 获取用户的实际交互物品
                true_items = generator.user_interacted_items[users[user_idx].item()]

                # 计算DCG
                dcg = 0
                for i, item_idx in enumerate(indices[user_idx]):
                    if item_idx.item() in true_items:
                        dcg += 1 / np.log2(i + 2)

                # 计算IDCG
                idcg = 0
                for i in range(min(len(true_items), config['topk'])):
                    idcg += 1 / np.log2(i + 2)

                # 计算NDCG
                ndcg = dcg / idcg if idcg > 0 else 0
                ndcg_scores.append(ndcg)

    avg_ndcg = np.mean(ndcg_scores)
    return avg_ndcg

if __name__ == '__main__':
    with open("/graduation_design/bpr/config/bpr.yaml", 'r', encoding='utf-8') as f:
        config = yaml.unsafe_load(f)
    print(config)

    # 初始化样本生成器
    generator = BPRSampleGenerator(config['data'])
    seq_samples = generator.generate_seq_samples(
        seq_len=config['data']['maxlen'],
        num_negatives=config['data']['num_negatives']
    )

    dataset = SeqBPRDataset(seq_samples, config['model']['device'])
    train_size = int(config['data']['train_valid_split'] * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=True)

    criterion = BPRLoss()
    model = SeqLearn(config['model'], config['data'], generator.n_user, generator.n_item)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['model']['lr'])

    # train(config, model, train_loader, criterion, optimizer)

    # 加载最佳模型
    model.load_state_dict(torch.load(f"/graduation_design/bpr/ckpt/bpr_epoch1_batch60.pth"))

    avg_ndcg = test(config, model, test_loader)
    print(f"测试集上的平均NDCG@{config['topk']}: {avg_ndcg:.4f}")

    # torch.save(model.state_dict(), f"ckpt/bpr_ndcg{avg_ndcg:.4f}.pth")
