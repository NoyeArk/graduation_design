import numpy as np
import pandas as pd


class Data:
    """
    数据处理类, 用于处理数据
    """
    def __init__(self):
        self.users = pd.read_csv('data/ml-1m/users.dat', sep='::',
                                names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'],
                                engine='python')
        self.movies = pd.read_csv('data/ml-1m/movies.dat', sep='::',
                                 names=['MovieID', 'Title', 'Genres'],
                                 engine='python')
        self.ratings = pd.read_csv('data/ml-1m/ratings.dat', sep='::',
                                  names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
                                  engine='python')

        # 将所有 ID 减 1
        self.users['UserID'] = self.users['UserID'] - 1
        self.movies['MovieID'] = self.movies['MovieID'] - 1
        self.ratings['UserID'] = self.ratings['UserID'] - 1 
        self.ratings['MovieID'] = self.ratings['MovieID'] - 1

        self.num_users = self.users['UserID'].max() + 1
        self.num_items = self.movies['MovieID'].max() + 1
        self.num_ratings = len(self.ratings)

        print('user count:', self.num_users)
        print('item count:', self.num_items)

        # 获取每个用户评价过的电影字典(按时间排序)
        self.user_items = {}
        for user_id in self.ratings['UserID'].unique():
            user_ratings = self.ratings[self.ratings['UserID'] == user_id].sort_values('Timestamp')
            item_ids = user_ratings['MovieID'].tolist()
            self.user_items[user_id] = item_ids

        # 获取每个电影被评价过的用户字典
        self.item_users = {}
        for item_id in self.ratings['MovieID'].unique():
            item_ratings = self.ratings.query('MovieID == @item_id').sort_values('Timestamp')
            user_ids = item_ratings['UserID'].tolist()
            self.item_users[item_id] = user_ids

        # 划分训练集和验证集
        self.train_data, self.valid_data = self._split_train_valid()

    def _split_train_valid(self):
        """
        划分训练集和测试集
        将每个用户的交易数据按时间顺序划分, 前80%作为训练集, 后20%作为测试集

        Returns:
            train_data (`dict`): 训练集数据, key为用户ID, value为该用户的电影ID列表
            valid_data (`dict`): 验证集数据, key为用户ID, value为该用户的电影ID列表
        """
        train_data = {}
        valid_data = {}

        for user_id, items_id in self.user_items.items():
            split_point = int(len(items_id) * 0.8)
            train_data[user_id] = items_id[:split_point]
            valid_data[user_id] = items_id[split_point:]

        return train_data, valid_data

    def sample_negative_item(self, user_id):
        """
        采样负样本
        
        Args:
            user_id: 用户ID
            
        Returns:
            int: 采样的物品ID
        """
        if user_id not in self.user_items:
            return np.random.choice(self.num_items)

        items_id = self.user_items[user_id]
        return np.random.choice([i for i in range(self.num_items) if i not in items_id])
