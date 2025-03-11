import yaml
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

# from model.seq_learn import SeqLearn
from data import BPRSampleGenerator, BPRDataset, BPRLoss

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
# 划分训练集和测试集
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

for batch in dataloader:
    print(batch)
    break

print(seq_samples)
