import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset


class BPRSampleGenerator:
    """
    贝叶斯个性化排序(BPR)样本生成器
    用于从用户-物品交互数据中构造正负样本对
    """

    def __init__(self, args):
        """
        初始化BPR样本生成器
        
        Args:
            data_path: 数据文件路径，如果为None则需要后续调用load_data方法
            sep: 数据分隔符
            names: 列名列表，默认为['user', 'item', 'rating', 'timestamp']
            rating_threshold: 评分阈值，大于等于该值的交互被视为正向交互
        """
        self.data = None
        self.args = args
        self.user_interacted_items = None
        self.all_items = None
        self.user_to_id = {}
        self.item_to_id = {}
        self.id_to_user = {}
        self.id_to_item = {}

        self.names = ['user', 'item', 'rating', 'timestamp']
        self.rating_threshold = args['rating_threshold']
        self.user_threshold = args['user_threshold']
        self.item_threshold = args['item_threshold']
        self.load_data(args['path'])

    def load_data(self, data_path):
        """
        加载数据并进行预处理
        
        Args:
            data_path: 数据文件路径
            sep: 数据分隔符
        """
        # 加载数据
        self.data = pd.read_csv(data_path, sep='::', header=None, engine='python')
        self.data.columns = ['user', 'item', 'rating', 'timestamp']

        # 过滤评分
        if self.rating_threshold > 0:
            self.data = self.data[self.data['rating'] > self.rating_threshold]
            
        # 过滤交互次数少于阈值的用户和物品
        user_counts = self.data['user'].value_counts()
        item_counts = self.data['item'].value_counts()
        
        users_to_keep = user_counts[user_counts >= self.user_threshold].index
        items_to_keep = item_counts[item_counts >= self.item_threshold].index
    
        self.data = self.data[self.data['user'].isin(users_to_keep)]
        self.data = self.data[self.data['item'].isin(items_to_keep)]

        # 重置索引
        self.data = self.data.reset_index(drop=True)

        # 用户和物品ID映射（如果原始ID不是连续整数）
        self._create_id_mappings()

        # 检查交互索引的范围
        if self.interaction_indices:
            min_idx = min(self.interaction_indices.values())
            max_idx = max(self.interaction_indices.values()) 
            print(f">>>> 交互索引范围: 最小值 = {min_idx}, 最大值 = {max_idx}")

        # 按时间戳排序
        self.data = self.data.sort_values('timestamp')

        # 构建用户交互字典
        self.user_interacted_items = defaultdict(set)
        for row in self.data.itertuples():
            user_id = self.user_to_id[row.user]
            item_id = self.item_to_id[row.item]
            self.user_interacted_items[user_id].add(item_id)

        # 所有物品集合
        self.all_items = set(self.data['item_id'].unique())
        
        print(f">>>> 数据加载完成: {len(self.data)} 条交互, {len(self.user_to_id)} 个用户, {len(self.item_to_id)} 个物品")

        # 加载基模型的预测结果
        self.base_model_preds = []
        for base_model in self.args['base_model']:
            res = np.load(self.args['base_model_path'] + f"/{base_model}.npy")
            self.base_model_preds.append(res)
        self.base_model_preds = np.stack(self.base_model_preds, axis=1)
        print(f">>>> 基模型的预测结果加载完成: {self.base_model_preds.shape}")

    def _create_id_mappings(self):
        """创建用户和物品的ID映射"""
        # 获取唯一用户和物品
        unique_users = self.data['user'].unique()
        unique_items = self.data['item'].unique()

        # 创建映射字典
        self.user_to_id = {user: i for i, user in enumerate(unique_users)}
        self.item_to_id = {item: i for i, item in enumerate(unique_items)}
        self.id_to_user = {i: user for user, i in self.user_to_id.items()}
        self.id_to_item = {i: item for item, i in self.item_to_id.items()}

        # 添加ID列到数据中
        self.data['user_id'] = self.data['user'].map(self.user_to_id)
        self.data['item_id'] = self.data['item'].map(self.item_to_id)

        # 添加交互索引映射
        self.interaction_indices = {}
        for idx, row in self.data.iterrows():
            key = (row['user_id'], row['item_id'])
            self.interaction_indices[key] = idx

    def get_user_seq(self, max_seq_len=50):
        """
        获取每个用户的历史交互序列

        Args:
            max_seq_len: 序列最大长度，默认为50
            
        Returns:
            user_seq: 字典，键为用户ID，值为该用户按时间排序的交互物品ID列表
        """
        user_seq = defaultdict(list)
        
        # 按用户ID和时间戳排序
        sorted_data = self.data.sort_values(['user_id', 'timestamp'])
        
        for row in sorted_data.itertuples():
            user_id = row.user_id
            item_id = row.item_id
            user_seq[user_id].append(item_id)
            
            # 如果序列超过最大长度，保留最近的max_seq_len个交互
            if len(user_seq[user_id]) > max_seq_len:
                user_seq[user_id] = user_seq[user_id][-max_seq_len:]
                
        print(f">>>> 构建了 {len(user_seq)} 个用户的历史交互序列")
        return user_seq
    
    def generate_basic_samples(self, num_negatives=1):
        """
        生成基本的BPR训练样本
        
        Args:
            num_negatives: 每个正样本对应的负样本数量
            
        Returns:
            samples: 包含(user_id, pos_item_id, neg_item_id)三元组的列表
        """
        if self.data is None:
            raise ValueError("请先加载数据")
            
        samples = []
        for row in self.data.itertuples():
            user_id = row.user_id
            pos_item_id = row.item_id
            
            # 为每个正样本采样指定数量的负样本
            for _ in range(num_negatives):
                # 从用户未交互过的物品中随机选择一个作为负样本
                neg_items = list(self.all_items - self.user_interacted_items[user_id])
                if not neg_items:  # 如果用户交互了所有物品（极少见）
                    continue
                    
                neg_item_id = np.random.choice(neg_items)
                samples.append((user_id, pos_item_id, neg_item_id))

        print(f">>>> 生成了 {len(samples)} 个BPR样本对")
        return samples
    
    def generate_seq_samples(self, seq_len=10, num_negatives=1):
        """
        生成包含用户历史序列的BPR训练样本
        
        Args:
            seq_len: 历史序列长度
            num_negatives: 每个正样本对应的负样本数量
            
        Returns:
            samples: 包含(user_id, history_seq, pos_item_id, neg_item_id)的列表，
                    以及基模型预测结果
        """
        if self.data is None:
            raise ValueError("请先加载数据")
            
        samples = []
        # 获取用户序列
        user_seq = self.get_user_seq(max_seq_len=seq_len)  # 获取足够长的序列以便截取

        # 按用户和时间戳分组
        user_data = self.data.sort_values(['user_id', 'timestamp'])
        
        for user_id, group in tqdm(user_data.groupby('user_id'), desc="生成序列样本"):
            items = group['item_id'].tolist()

            # 对于每个交互，使用其之前的seq_len个交互作为历史序列
            for i in range(1, len(items)):
                pos_item_id = items[i]
                interaction_idx = self.get_interaction_index(user_id, pos_item_id)

                # 获取当前交互的基模型预测结果
                if interaction_idx is not None:
                    base_model_preds = self.base_model_preds[interaction_idx, :, :100]  # [k, 100]
                else:
                    base_model_preds = None

                # 为每个正样本采样负样本
                for _ in range(num_negatives):
                    neg_items = list(self.all_items - self.user_interacted_items[user_id])
                    if not neg_items:
                        continue

                    neg_item_id = np.random.choice(neg_items)
                    samples.append((user_id, user_seq[user_id][:seq_len], pos_item_id, neg_item_id, base_model_preds))

        print(f">>>> 生成了 {len(samples)} 个序列感知BPR样本对")
        return samples

    def generate_time_aware_samples(self, num_negatives=1):
        """
        生成时间感知的BPR训练样本
        
        Args:
            num_negatives: 每个正样本对应的负样本数量
            
        Returns:
            samples: 包含(user_id, pos_item_id, neg_item_id)三元组的列表
        """
        if self.data is None:
            raise ValueError("请先加载数据")
            
        samples = []
        # 按用户和时间戳分组
        user_data = self.data.sort_values(['user_id', 'timestamp'])
        
        for user_id, group in user_data.groupby('user_id'):
            items = group['item_id'].tolist()
            timestamps = group['timestamp'].tolist()
            
            # 使用较新的交互作为正样本
            for i in range(len(items)):
                pos_item_id = items[i]
                
                # 为每个正样本采样负样本
                for _ in range(num_negatives):
                    neg_items = list(self.all_items - self.user_interacted_items[user_id])
                    if not neg_items:
                        continue
                        
                    neg_item_id = np.random.choice(neg_items)
                    samples.append((user_id, pos_item_id, neg_item_id))
        
        print(f">>>> 生成了 {len(samples)} 个时间感知BPR样本对")
        return samples
    
    def generate_rating_aware_samples(self, num_negatives=1, top_n=5):
        """
        生成评分感知的BPR训练样本
        
        Args:
            num_negatives: 每个正样本对应的负样本数量
            top_n: 每个用户选取评分最高的前N个物品作为正样本
            
        Returns:
            samples: 包含(user_id, pos_item_id, neg_item_id)三元组的列表
        """
        if self.data is None:
            raise ValueError("请先加载数据")

        samples = []
        
        # 按评分降序排列
        user_data = self.data.sort_values(['user_id', 'rating'], ascending=[True, False])
        
        for user_id, group in user_data.groupby('user_id'):
            # 获取用户评分最高的几个物品作为正样本
            top_items = group['item_id'].tolist()[:top_n]
            
            for pos_item_id in top_items:
                # 为每个正样本采样负样本
                for _ in range(num_negatives):
                    neg_items = list(self.all_items - self.user_interacted_items[user_id])
                    if not neg_items:
                        continue
                        
                    neg_item_id = np.random.choice(neg_items)
                    samples.append((user_id, pos_item_id, neg_item_id))
        
        print(f">>>> 生成了 {len(samples)} 个评分感知BPR样本对")
        return samples
    
    def generate_hard_negative_samples(self, model, num_negatives=1, num_candidates=10):
        """
        生成难负样本的BPR训练样本
        
        Args:
            model: 推荐模型，需要有predict(user_id, item_id)方法
            num_negatives: 每个正样本对应的负样本数量
            num_candidates: 候选负样本数量
            
        Returns:
            samples: 包含(user_id, pos_item_id, neg_item_id)三元组的列表
        """
        if self.data is None:
            raise ValueError("请先加载数据")
            
        samples = []
        
        for row in self.data.itertuples():
            user_id = row.user_id
            pos_item_id = row.item_id
            
            # 为每个正样本采样多个候选负样本
            neg_items = list(self.all_items - self.user_interacted_items[user_id])
            if len(neg_items) < num_candidates:
                continue
                
            # 随机选择一些候选负样本
            candidates = np.random.choice(neg_items, num_candidates, replace=False)
            
            # 计算模型对这些候选的预测分数
            candidate_scores = [model.predict(user_id, item_id) for item_id in candidates]
            
            # 选择得分最高的几个作为难负样本
            sorted_indices = np.argsort(candidate_scores)[::-1]  # 降序排列
            hard_negatives = [candidates[i] for i in sorted_indices[:num_negatives]]
            
            for neg_item_id in hard_negatives:
                samples.append((user_id, pos_item_id, neg_item_id))

        print(f">>>> 生成了 {len(samples)} 个难负样本BPR样本对")
        return samples

    def get_interaction_index(self, user_id, item_id):
        """获取用户-物品交互在数据集中的索引"""
        key = (user_id, item_id)
        return self.interaction_indices.get(key, -1)


class BPRDataset(Dataset):
    def __init__(self, samples):
        """
        初始化BPR数据集
        
        Args:
            samples: 包含(user_id, pos_item_id, neg_item_id)三元组的列表
        """
        self.users = [sample[0] for sample in samples]
        self.user_seq = [sample[1] for sample in samples]
        self.pos_items = [sample[2] for sample in samples]
        self.neg_items = [sample[3] for sample in samples]
        self.base_model_preds = [sample[4] for sample in samples]

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.user_seq[idx], self.pos_items[idx], self.neg_items[idx], self.base_model_preds[idx]


class SeqBPRDataset(Dataset):
    """序列感知的BPR数据集类，用于PyTorch DataLoader"""

    def __init__(self, samples):
        """
        初始化序列感知的BPR数据集
        
        Args:
            samples: 包含(user_id, history_seq, pos_item_id, neg_item_id)的列表
        """
        self.users = [sample[0] for sample in samples]
        self.histories = [sample[1] for sample in samples]
        self.pos_items = [sample[2] for sample in samples]
        self.neg_items = [sample[3] for sample in samples]

    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return self.users[idx], self.histories[idx], self.pos_items[idx], self.neg_items[idx]


class BPRLoss(nn.Module):
    """BPR损失函数实现"""
    
    def __init__(self):
        super(BPRLoss, self).__init__()
        
    def forward(self, pos_scores, neg_scores):
        """
        计算BPR损失
        
        Args:
            pos_scores: 正样本得分，形状为 [batch_size]
            neg_scores: 负样本得分，形状为 [batch_size]
            
        Returns:
            loss: BPR损失值
        """
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        return loss
