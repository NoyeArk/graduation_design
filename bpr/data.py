import copy
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset


class Data(Dataset):
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
        self.user_interacted_item_ids = None
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

        # 划分训练集和测试集
        # split_index = int(len(self.data) * args['train_test_split'])
        # self.train_data = self.data[:split_index]
        # self.test_data = self.data[split_index:]

        # [len(train_data) * num_negatives, dict[7]]
        # train_samples, test_samples = self.generate_samples(seq_len=args['maxlen'], num_negatives=args['num_negatives'])
        # self.train_dataset = SeqBPRDataset(train_samples, args['device'])
        # self.test_dataset = SeqBPRDataset(test_samples, args['device'], is_test=True)

    def load_data(self, data_path):
        """
        加载数据并进行预处理

        Args:
            data_path: 数据文件路径
            sep: 数据分隔符
        """
        if 'nrows' in self.args:
            self.data = pd.read_csv(data_path, sep=self.args['sep'], header=None, engine='python', nrows=self.args['nrows'])
        else:
            self.data = pd.read_csv(data_path, sep=self.args['sep'], header=None, engine='python')
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

        self.data = self.data.sort_values(['user', 'timestamp'])
        self.data = self.data.reset_index(drop=True)

        # 用户和物品ID映射（如果原始ID不是连续整数）
        self._create_id_mappings()

        # 构建用户交互字典
        self.user_interacted_item_ids = defaultdict(list)
        for row in self.data.itertuples():
            self.user_interacted_item_ids[row.user_id].append(row.item_id)

        # 所有物品集合
        self.all_item_ids = set(self.data['item_id'].unique())

        print(f">>>> 数据加载完成: {len(self.data)} 条交互, {self.n_user} 个用户, {self.n_item} 个物品")

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

        self.n_user = len(self.user_to_id)
        self.n_item = len(self.item_to_id)

        # 添加ID列到数据中
        self.data['user_id'] = self.data['user'].map(self.user_to_id)
        self.data['item_id'] = self.data['item'].map(self.item_to_id)

        # 添加交互索引映射
        self.interaction_indices = {}
        for idx, row in self.data.iterrows():
            key = (row['user_id'], row['item_id'])
            self.interaction_indices[key] = idx

    def label_positive(self):
        """
        返回正样本得分的函数

        Returns:
            tuple: 包含用户-物品对和对应的得分
        """
        # 获取基础模型的数量
        n_base_models = len(self.args['base_model'])
        # 创建得分矩阵，形状为 [样本数, 基模型数]
        pos_label = np.zeros([len(self.data), n_base_models])
        # 获取真实 (Ground Truth) 物品 ID，扩展维度成 [batch, 1]
        gt_item = np.expand_dims(self.data['item_id'].values, axis=1)

        # 复制基模型预测的前 N 个排名结果，形状为 [batch, k, N]，k 是基模型数量，N 是考虑的排名数量
        rank_chunk = copy.deepcopy(self.base_model_preds[:, :, 2:2+self.args['base_model_topk']])  # [batch, k, rank]

        for k in range(n_base_models):
            # 获取当前基模型的排名结果
            rank_chunk_k = rank_chunk[:, k, :]
            # 比较真实物品是否在排名中
            is_item_in_rank = gt_item == rank_chunk_k
            # 如果物品在排名中，计算得分 = 1 / (10 + 物品的排名位置)
            pos_label[np.sum(is_item_in_rank, axis=1) > 0, k] = 1 / (10 + np.argwhere(is_item_in_rank)[:, 1])

        # 对应的得分
        return pos_label  # [n_sample, k]

    def label_negative(self, neg_item_ids):
        """
        返回负样本得分的函数

        Args:
            neg_item_ids (`np.ndarray`): 负样本列表

        Returns:
            label (`np.ndarray`): 负样本得分
        """
        n_base_models = len(self.args['base_model'])
        neg_label = np.zeros([len(neg_item_ids), n_base_models])  # [n_sample * num_neg, k]
        gt_item = np.expand_dims(neg_item_ids, axis=1)  # [n_sample * num_neg, 1]
        rank_chunk = copy.deepcopy(
            self.base_model_preds[:, :, 2:2+self.args['base_model_topk']]
        )  # [batch, k, topk]

        for k in range(n_base_models):
            rank_chunk_k = rank_chunk[:, k, :]
            torf = gt_item == rank_chunk_k
            neg_label[np.sum(torf, axis=1) > 0, k] = 1 / (10 + np.argwhere(torf)[:, 1])

        return neg_label

    def all_item_score(self, dataset):
        """
        返回所有得分物品的函数

        Args:
            dataset (`np.ndarray`): 训练集或测试集, [n_samples, k, 2+rank]

        Returns:
            u_k_i (`np.ndarray`): 所有得分, [n_samples, k, n_item]
        """
        # rank_chunk = dataset[:,:,2:2+self.args['base_model_topk']]  # [batch, k, rank]
        n_samples, k, topk = dataset.shape  # [batch, k, rank]
        rank_chunk_reshape = np.reshape(dataset, [-1, topk])

        u_k_i = np.zeros([n_samples * k, self.n_item], dtype=np.float32)  # [batch, k, n_item]
        for i in range(topk):
            u_k_i[np.arange(len(u_k_i)), rank_chunk_reshape[:, i]] = 1 / (i + 10)
        return np.reshape(u_k_i, [n_samples, k, self.n_item])

    def generate_samples(self, seq_len=10, num_negatives=1):
        """
        生成包含用户历史序列的BPR训练样本

        Args:
            seq_len: 历史序列长度
            num_negatives: 每个正样本对应的负样本数量
            
        Returns:
            samples: 包含(user_id, history_seq, pos_item_id, neg_item_id)的列表，
                    以及基模型预测结果
        """
        user_item_pairs = []  # [n_sample, 2]
        user_seq = []  # [n_sample, seq_len]
        pos_labels = self.label_positive()  # [n_sample, k]
        neg_labels = []  # [n_sample * num_negatives, k]
        base_model_preds = []  # [n_sample, k, 100]
        neg_samples = []  # [n_sample * num_negatives]

        user_data = self.data.sort_values(['user', 'timestamp'])

        for user_id, group in tqdm(user_data.groupby('user_id'), desc=">>>> 采样负样本"):
            item_ids = group['item_id'].tolist()  # 获取原始item id
            items = group['item'].tolist()  # 获取原始item

            # 预先获取用户未交互物品列表,避免重复计算
            neg_item_ids = list(self.all_item_ids - set(self.user_interacted_item_ids[user_id]))

            for item_id in item_ids:
                user_item_pairs.append([user_id, item_id])

                # 获取用户交互过的item之前的seq_len个交互作为历史序列
                item_idx = items.index(self.id_to_item[item_id])
                user_history = items[max(0, item_idx - seq_len):item_idx]

                # 如果历史序列长度不足seq_len，则在前面填充0
                if len(user_history) < seq_len:
                    user_history = user_history + [0] * (seq_len - len(user_history))
                user_seq.append(user_history)

                interaction_idx = self.get_interaction_index(user_id, item_id)
                assert (self.base_model_preds[interaction_idx, :, :2] == (user_id, item_id)).all()
                base_model_preds.append(self.base_model_preds[interaction_idx, :, 2:2+seq_len])  # [k, seq_len]

                # 为每个正样本采样负样本
                sample_neg_item_ids = np.random.choice(neg_item_ids, size=num_negatives, replace=False)
                neg_samples.extend(sample_neg_item_ids)

        neg_labels = self.label_negative(neg_samples)

        train_samples, test_samples = [], []
        train_size = int(len(user_item_pairs) * self.args['train_test_split'])
        test_size = len(user_item_pairs) - train_size

        base_model_preds = np.array(base_model_preds).astype(np.int32)
        base_model_preds_test = base_model_preds[-test_size:]
        all_scores = self.all_item_score(base_model_preds_test)

        # 将 base_model_preds 的索引从 [0, 3122] 转换为 [1, 3952]
        vectorized_id_map = np.vectorize(lambda x: self.id_to_item[x])
        base_model_preds = vectorized_id_map(base_model_preds)

        for i in tqdm(range(train_size), desc=">>>> 构建训练集"):
            train_samples.append({
                'user_id': user_item_pairs[i][0],  # 用户ID
                'history_seq': user_seq[i],  # [seq_len]
                'pos_item': self.id_to_item[user_item_pairs[i][1]],  # 正样本
                'neg_item': self.id_to_item[neg_samples[i]],  # 负样本
                'pos_label': pos_labels[i],  # [k]
                'neg_label': neg_labels[i],  # [k]
                'base_model_preds': base_model_preds[i]  # [k, seq_len]
            })
        print(f">>>> 生成了 {len(train_samples)} 个训练样本")

        for j in tqdm(range(train_size, len(user_item_pairs)), desc=">>>> 构建测试集"):
            test_samples.append({
                'user_id': user_item_pairs[j][0],  # 用户ID
                'history_seq': user_seq[j],  # [seq_len]
                'pos_item': self.id_to_item[user_item_pairs[j][1]],  # 正样本
                'neg_item': self.id_to_item[neg_samples[j]],  # 负样本
                'all_item_scores': all_scores[j - train_size],  # [k, all_items]
                'base_model_preds': base_model_preds[j]  # [k, seq_len]
            })
        print(f">>>> 生成了 {len(test_samples)} 个测试样本")

        return train_samples, test_samples

    def get_interaction_index(self, user_id, item_id):
        """获取用户-物品交互在数据集中的索引"""
        key = (user_id, item_id)
        return self.interaction_indices.get(key, -1)


class SeqBPRDataset(Dataset):
    def __init__(self, samples, device, is_test=False):
        """
        初始化BPR数据集
        
        Args:
            samples: 包含(user_id, seq_len, pos_item_id, neg_item_id, base_model_preds)五元组的列表
            user_id: 形状为[batch_size]
            seq_len: 形状为[batch_size, seq_len]
            pos_item_id: 形状为[batch_size]
            neg_item_id: 形状为[batch_size]
            pos_label: 形状为[batch_size, k]
            neg_label: 形状为[batch_size, k]
            base_model_preds: 形状为[batch_size, k, 100]
        """
        self.users = [sample['user_id'] for sample in samples]
        self.histories = [sample['history_seq'] for sample in samples]
        self.pos_items = [sample['pos_item'] for sample in samples]
        self.neg_items = [sample['neg_item'] for sample in samples]
        if not is_test:
            self.pos_labels = [sample['pos_label'] for sample in samples]
            self.neg_labels = [sample['neg_label'] for sample in samples]
        else:
            self.all_item_scores = [sample['all_item_scores'] for sample in samples]
        self.base_model_preds = [sample['base_model_preds'] for sample in samples]

        self.device = device
        self.is_test = is_test

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        case = {
            'user_id': torch.tensor(self.users[idx], device=self.device),
            'user_seq': torch.tensor(self.histories[idx], device=self.device),
            'pos_item': torch.tensor(self.pos_items[idx], device=self.device),
            'neg_item': torch.tensor(self.neg_items[idx], device=self.device),
            'base_model_preds': torch.tensor(self.base_model_preds[idx], device=self.device)
        }
        if self.is_test:
            case['all_item_scores'] = torch.tensor(self.all_item_scores[idx], device=self.device)
        else:
            case['pos_label'] = torch.tensor(self.pos_labels[idx], device=self.device)
            case['neg_label'] = torch.tensor(self.neg_labels[idx], device=self.device)
        return case
