import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from model.seq_learn import SeqLearn
from data import BPRSampleGenerator, BPRDataset, BPRLoss


if __name__ == '__main__':
    with open("/root/autodl-tmp/graduation_design/rec/config/bpr.yaml", 'r', encoding='utf-8') as f:
        config = yaml.unsafe_load(f)
    print(config)

    # 初始化样本生成器
    generator = BPRSampleGenerator(config['data'])
    seq_samples = generator.generate_seq_samples(
        seq_len=config['data']['maxlen'],
        num_negatives=config['data']['num_negatives']
    )

    dataset = BPRDataset(seq_samples)
    train_size = int(config['data']['train_valid_split'] * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    criterion = BPRLoss()

    model = SeqLearn(config['model'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['model']['lr'])
    
    for epoch in range(config['epoch']):
        total_loss = 0

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epoch']}") as pbar:
            for batch in pbar:
                users, user_seq, pos_items, neg_items, base_model_preds = batch
                optimizer.zero_grad()
                pos_scores, neg_scores = model(users, user_seq, pos_items, neg_items, base_model_preds)
                loss = criterion(pos_scores, neg_scores)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix(loss=f"{total_loss/(pbar.n+1):.4f}")

    model.eval()
    with torch.no_grad():
        ndcg_scores = []
        for batch in test_loader:
            users, user_seq, pos_items, neg_items, base_model_preds = batch

            # 获取所有物品的预测分数
            all_items = torch.arange(len(generator.item_to_id)).to(config['model']['device'])
            all_scores = []
            
            # 分批次预测所有物品的分数以避免内存溢出
            batch_size = 1024
            for i in range(0, len(all_items), batch_size):
                batch_items = all_items[i:i+batch_size]
                batch_users = users.unsqueeze(1).repeat(1, len(batch_items))
                batch_seqs = user_seq.unsqueeze(1).repeat(1, len(batch_items), 1)
                scores = model.predict(batch_users, batch_seqs, batch_items)
                all_scores.append(scores)

            all_scores = torch.cat(all_scores, dim=1)
            
            # 计算每个用户的NDCG@k
            k = 10
            _, indices = torch.topk(all_scores, k)
            
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
                for i in range(min(len(true_items), k)):
                    idcg += 1 / np.log2(i + 2)
                    
                # 计算NDCG
                ndcg = dcg / idcg if idcg > 0 else 0
                ndcg_scores.append(ndcg)

    avg_ndcg = np.mean(ndcg_scores)
    print(f"测试集上的平均NDCG@{k}: {avg_ndcg:.4f}")

    torch.save(model.state_dict(), config['model']['path'])
