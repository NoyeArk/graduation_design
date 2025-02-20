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

        # 将所有ID减1
        self.users['UserID'] = self.users['UserID'] - 1
        self.movies['MovieID'] = self.movies['MovieID'] - 1
        self.ratings['UserID'] = self.ratings['UserID'] - 1 
        self.ratings['MovieID'] = self.ratings['MovieID'] - 1

        self.num_users = self.users['UserID'].max() + 1
        self.num_movies = self.movies['MovieID'].max() + 1
        self.num_ratings = len(self.ratings)

        print('user count:', self.num_users)
        print('movie count:', self.num_movies)

        # 获取每个用户评价过的电影字典(按时间排序)
        self.user_movies = {}
        for user_id in self.ratings['UserID'].unique():
            user_ratings = self.ratings[self.ratings['UserID'] == user_id].sort_values('Timestamp')
            movie_ids = user_ratings['MovieID'].tolist()
            self.user_movies[user_id] = movie_ids

        # 获取每个电影被评价过的用户字典
        self.movie_users = {}
        for movie_id in self.ratings['MovieID'].unique():
            movie_ratings = self.ratings[self.ratings['MovieID'] == movie_id].sort_values('Timestamp')
            user_ids = movie_ratings['UserID'].tolist()
            self.movie_users[movie_id] = user_ids

        # 划分训练集和验证集
        self.train_data, self.valid_data = self.split_train_valid()

    def split_train_valid(self):
        """
        划分训练集和测试集
        将每个用户的交易数据按时间顺序划分, 前80%作为训练集, 后20%作为测试集
        
        Returns:
            train_data (`dict`): 训练集数据, key为用户ID, value为该用户的电影ID列表
            valid_data (`dict`): 验证集数据, key为用户ID, value为该用户的电影ID列表
        """
        train_data = {}
        valid_data = {}

        for user_id, movie_list in self.user_movies.items():
            split_point = int(len(movie_list) * 0.8)
            train_data[user_id] = movie_list[:split_point]
            valid_data[user_id] = movie_list[split_point:]

        return train_data, valid_data
