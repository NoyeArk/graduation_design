import yaml
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import SeqLearn
from torch.utils.tensorboard import SummaryWriter
from data import BPRSampleGenerator, SeqBPRDataset


def train(config, model, train_loader, optimizer):
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
                users, user_seq, items, scores, base_model_preds = batch
                optimizer.zero_grad()
                preds = model(users, user_seq, items, base_model_preds)
                loss = torch.sqrt(torch.mean((preds - scores) ** 2))
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
                    preds=f"{preds.mean().item():.4f}",
                    scores=f"{scores.float().mean().item():.4f}"
                )

                # 记录各项指标
                train.writer.add_scalar('训练/损失', loss.item(), train.global_step)
                
                # 更新全局步数
                train.global_step += 1

                # 每训练5个batch保存一次模型
                if (batch_idx + 1) % 300 == 0:
                    torch.save(model.state_dict(), f"/root/autodl-tmp/score_ckpt/score_epoch{epoch+1}_batch{batch_idx+1}.pth")
                    print(f"模型已保存: ckpt/score_epoch{epoch+1}_batch{batch_idx+1}.pth")

                    # avg_ndcg = test(config, model, test_loader)
                    # print(f"测试集上的平均NDCG@{config['topk']}: {avg_ndcg:.4f}")

                    # train.writer.add_scalar(f'测试/NDCG@{config["topk"]}', avg_ndcg, train.global_step)


def test(config, model, test_loader):
    model.eval()
    with torch.no_grad():
        ndcg_scores = []

        # 将测试集重置到初始状态
        test_loader.dataset.dataset.reset()
        for batch in tqdm(test_loader, desc="计算测试集NDCG"):
            users, user_seq, items, scores, base_model_preds = batch

            # 对每个用户计算NDCG
            for user_idx in range(len(users)):
                # 获取当前用户
                user = users[user_idx:user_idx+1]
                user_seq_i = user_seq[user_idx:user_idx+1]
                base_model_preds_i = base_model_preds[user_idx:user_idx+1] if base_model_preds is not None else None

                # 获取所有物品ID
                all_item_ids = torch.arange(generator.n_item, device=config['model']['device'])
                
                all_scores = []
                
                for i in range(0, len(all_item_ids), config['batch_size']):
                    batch_items = all_item_ids[i:i + config['batch_size']]
                    cnt = len(batch_items)

                    user_repeated = user.repeat(cnt)
                    user_seq_repeated = user_seq_i.repeat(cnt, 1)
                    base_model_preds_repeated = base_model_preds_i.repeat(cnt, 1, 1)
                    # 预测分数
                    batch_scores = model(user_repeated, user_seq_repeated, batch_items, base_model_preds_repeated)
                    all_scores.append(batch_scores)

                # 合并所有批次的分数
                user_scores = torch.cat(all_scores, dim=0).squeeze()
                
                # 获取前k个物品
                _, indices = torch.topk(user_scores, config['topk'])

                # 获取用户的实际交互物品
                true_items = generator.user_interacted_items[user.item()]

                # 计算DCG
                dcg = 0
                for i, item_idx in enumerate(indices):
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
    model.train()  # 恢复训练模式
    return avg_ndcg


if __name__ == '__main__':
    with open("/graduation_design/score/score.yaml", 'r', encoding='utf-8') as f:
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

    model = SeqLearn(config['model'], config['data'], generator.n_user, generator.n_item)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['model']['lr'])

    train(config, model, train_loader, optimizer)

    # 加载最佳模型
    # model.load_state_dict(torch.load(f"/root/autodl-tmp/ckpt/bpr_epoch1_batch90.pth"))

    # avg_ndcg = test(config, model, test_loader)
    # print(f"测试集上的平均NDCG@{config['topk']}: {avg_ndcg:.4f}")

    # torch.save(model.state_dict(), f"score_ckpt/score_ndcg{avg_ndcg:.4f}.pth")
