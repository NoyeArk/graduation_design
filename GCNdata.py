import copy
import codecs
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix, save_npz

leave_num = 1000000000
remove_rating = 3.5
last = 10  # 5 for BMS; 10 for SNR


def create_relation_dict(entity_count_1, entity_count_2, user_item_pairs):
    """
    将用户-物品对转化为关系字典

    Args:
        entity_count_1 (`int`): 实体 1 的数量
        entity_count_2 (`int`): 实体 2 的数量
        user_item_pairs (`list`): 用户-物品对

    Returns:
        `dict`: 包含实体 1 到实体 2 的映射关系的字典。
    """
    # 将总数 entity_count_1，entity_count_2 的 entity 转化成映射字典
    forward_dict = {i: [] for i in range(entity_count_1)}
    reverse_dict = {i: [] for i in range(entity_count_2)}
    for user, item in user_item_pairs:
        if item not in forward_dict[user]:
            forward_dict[int(user)].append(int(item))
            reverse_dict[int(item)].append(int(user))
    return forward_dict


def remove_unrating_user_and_rename(user2id, item2id, user_item):
    """
    去掉没有的 user 和 item 对应的 rating, 并 rename

    Args:
        user2id (`dict`): 用户 ID 映射
        item2id (`dict`): 物品 ID 映射
        user_item (`list`): 用户-物品对

    Returns:
        list: 用户-物品对
    """
    res = []
    for user, item in user_item:
        if user in user2id.keys() and item in item2id.keys():
            res.append([user2id[user], item2id[item]])
    return res


def map_elements_to_id(elements):
    """
    将列表中的元素映射到 ID。

    Args:
        elements (`list`): 列表

    Returns:
        dict: 映射字典
    """
    return {v: k for k, v in enumerate(elements)}


def set_forward(dict1):
    return_1 = dict()
    for key in dict1.keys():
        sub = dict1[key]
        return_1[key] = { ky: set(sub[ky]) for ky in sub.keys()}
    return return_1


def build_sparse_matrix(user_num, item_num, user_item):
    """
    构建用户-物品的稀疏矩阵

    Args:
        user_num (`int`): 用户数量
        item_num (`int`): 物品数量
        user_item (`list`): 用户-物品对

    Returns:
        lil_matrix: 稀疏矩阵
    """
    res = lil_matrix((user_num, item_num))
    for user in user_item.keys():
        for item in user_item[user]:
            res[user, item] =1
    return res


class Data(object):
    """
    数据类
    """
    def __init__(self, args, seed=0, Markov=False):
        self.name_id = dict()
        self.name = args.dataset
        self.dir = args.path if args.path[-1] == '/' else args.path + '/'
        if args.name in ['CiaoDVD', 'dianping']:
            self.encoding = 'utf-8'
        else:
            self.encoding= 'iso-8859-15'
        self.epoch = 100
#        if self.name in ['ML']:
#            self.entity = ['user','item','G','A','C','D']#
#            self.user_side_entity = []
#            self.item_side_entity = ['G','A','C','D']#
#            self.drop_user,self.drop_item = 10,10
#        if self.name in ['CiaoDVD']:
#            self.entity = ['user','item','G']#'C',
#            self.user_side_entity = []
#            self.item_side_entity = ['G']#'C'
#            self.drop_user,self.drop_item = 3,3
        if self.name in ['Amazon_App', 'Games', 'ml1m']:
            self.entity = ['user','item','G']#'C',
            self.user_side_entity = []
            self.item_side_entity = ['G']#'C'
            self.drop_user, self.drop_item = 10, 10
        if self.name in ['Grocery']:
            self.entity = ['user','item','G']#'C',
            self.user_side_entity = []
            self.item_side_entity = ['G']#'C'
            self.drop_user,self.drop_item = 20, 20
        if self.name in [
            'Sport',
            'Clothing', 
            'Instant_Video',
            'Pet_Supplies',
            'Patio',
            'Office_Products',
            'Musical_Instruments', 
            'Digital_Music',
            'Baby',
            'Automotive'
        ]:
            self.entity = ['user','item','G']#'C',
            self.user_side_entity = []
            self.item_side_entity = ['G']#'C'
            self.drop_user, self.drop_item = 5,5
        if self.name in ['dianping']:
            self.entity = ['user','item','S','A','C']#
            self.user_side_entity = []
            self.item_side_entity = ['S','A','C']#'C'
            self.drop_user, self.drop_item = 20, 20  #之前是25, 50
        if self.name in ['Kindle', 'tiktok']:
            self.entity = ['user', 'item', 'G']#'C',
            self.user_side_entity = []
            self.item_side_entity = ['G']  #'C'
            self.drop_user, self.drop_item = 50, 50  #之前是50,50

        # save
        self.dict_entity2id = dict()  # entity -> ID
        self.dict_list = dict()  # entity1_entity2 -> 两元组
        self.dict_forward = dict()  # entity1_entity2 -> 关系字典
        self.dict_reverse = dict()
        self.entity_num = dict()  # 数量
        # read rating and split train and test set

        # 移除交互次数少于5次的用户和物品
        self.dict_list['user_item'], self.entity_num['user'], self.entity_num['item'], \
        self.dict_entity2id['user'], self.dict_entity2id['item'] = self.filter_user_item_by_rating(
            rating_threshold=remove_rating,
            user_threshold=self.drop_user,
            item_threshold=self.drop_item
        )  # 53

        # 获取每个用户最近的交互记录
        self.latest_interaction = self.get_latest_interaction(
            user_item_pairs=self.dict_list['user_item'],
            keep=args.maxlen
        )  # 时间特征

        # 分割训练、验证和测试集
        self.train_set, self.valid_set, self.test_set = self.split_dataset(
            user_item=self.dict_list['user_item']
        )

        # 时序特征
        self.dict_forward['train'] = create_relation_dict(
            entity_count_1=self.entity_num['user'],
            entity_count_2=self.entity_num['item'],
            user_item_pairs=self.train_set
        )
        self.dict_forward['valid'] = create_relation_dict(
            entity_count_1=self.entity_num['user'],
            entity_count_2=self.entity_num['item'],
            user_item_pairs=self.valid_set
        )
        self.dict_forward['test'] = create_relation_dict(
            entity_count_1=self.entity_num['user'],
            entity_count_2=self.entity_num['item'],
            user_item_pairs=self.test_set
        )
        self.markov = self.constrcut_markov_matrices()

        # 将物品-其他实体对转化为关系字典
        for entity in self.item_side_entity:
            self.dict_list['item_' + entity], \
            self.dict_entity2id[entity], \
            self.entity_num[entity] = self.get_item_entity_pairs('I' + entity + '.data')
            self.dict_forward['item_' + entity] = create_relation_dict(
                entity_count_1=self.entity_num['item'],
                entity_count_2=self.entity_num[entity],
                user_item_pairs=self.dict_list['item_' + entity]
            )
        for entity in self.user_side_entity:
            self.dict_list['user_' + entity], \
            self.dict_entity2id[entity], \
            self.entity_num[entity] = self.get_u_other('U' + entity + '.data')
            self.dict_forward['user_' + entity] = create_relation_dict(
                entity_count_1=self.entity_num['user'],
                entity_count_2=self.entity_num[entity],
                user_item_pairs=self.dict_list['user_' + entity]
            )

        # 构建实体-实体的稀疏矩阵
        # user-item
        self.matrix = dict()
        self.matrix['user_item'] = build_sparse_matrix(
            self.entity_num['user'],
            self.entity_num['item'],
            self.dict_forward['train']
        )
        self.matrix['item_user'] = self.matrix['user_item'].transpose()

        # 构建用户-其他实体的稀疏矩阵
        for entity in self.user_side_entity:
            self.matrix['user' + entity] = build_sparse_matrix(
                self.entity_num['user'],
                self.entity_num[entity],
                self.dict_forward['user_' + entity]
            )
            self.matrix[entity + 'user'] = self.matrix['user' + entity].transpose()

        # 构建物品-其他实体的稀疏矩阵
        for entity in self.item_side_entity:
            self.matrix['item' + entity] = build_sparse_matrix(
                self.entity_num['item'],
                self.entity_num[entity],
                self.dict_forward['item_' + entity]
            )
            self.matrix[entity + 'item'] = self.matrix['item' + entity].transpose()
        # self.set_forward = dict()
        # for key in self.dict_forward.keys():
        self.set_forward = set_forward(self.dict_forward)
        print(
            f'user: {self.entity_num["user"]}\t'
            f'item: {self.entity_num["item"]}\t'
            f'train: {len(self.train_set)}'
        )
        if self.name in ['Grocery', 'Kindle', 'Games']:
            self.pic = self.pic_feature('feature.npy')
        if self.name in ['tiktok']:
            self.pic = self.pic_feature('visual.npy')
            self.acou = self.pic_feature('acoustic.npy')

    def split_dataset(self, user_item, ratio=[0.6, 0.8]):
        """
        分割训练、验证和测试集

        Args:
            user_item (`list`): 用户-物品对
            ratio (`list`): 分割比例

        Returns:
            train_set (`list`): 训练集
            valid_set (`list`): 验证集
            test_set (`list`): 测试集
        """
        user_count = Counter(user_item[:, 0])
        train_set, valid_set, test_set = [], [], []

        n = 0
        for i, (user, item) in enumerate(user_item):
            if n < min(int(user_count[user] * ratio[0]), user_count[user] - 4):
                train_set.append([user, item])
                n = n + 1
            elif n < user_count[user] - 1:
                valid_set.append([user, item])
                n = n + 1
            else:
                test_set.append([user, item])
            try:
                if user_item[i + 1][0] != user:
                    n = 0
            except IndexError:
                pass

        return train_set, valid_set, test_set

    def get_latest_interaction(self, user_item_pairs, keep=last):
        """
        获取每个用户最近的交互记录

        Args:
            user_item_pairs (`list`): 用户-物品对
            keep (`int`): 保留的交互次数

        Returns:
            dict: { (user, item): [最近的交互记录] }
        """
        result = dict()
        init = [-1 for i in range(keep)]
        latest = copy.deepcopy(init)

        # 遍历用户-物品对
        for i, (user, item) in enumerate(user_item_pairs):
            result[(user, item)] = copy.deepcopy(list(latest[-keep:]))
            if i < len(user_item_pairs) - 1:
                if user_item_pairs[i + 1][0] != user:
                    latest = copy.deepcopy(init)
                else:
                    latest.append(item)
        return result

    def filter_user_item_by_rating(
        self,
        rating_threshold,
        user_threshold,
        item_threshold
    ):
        """
        获取数据集

        Args:
            rating_threshold (`float`): 评分阈值
            user_threshold (`int`): 用户评分阈值
            item_threshold (`int`): 物品评分阈值
            keep_interaction (`int`): 保留的交互次数
        
        Returns:
            user_item (`list`): 用户-物品对
            num_user (`int`): 用户数量
            num_item (`int`): 物品数量
            user2id (`dict`): 用户 ID 映射
            item2id (`dict`): 物品 ID 映射
        """
        file_path = self.dir + 'ratings.data'
        df = pd.read_csv(file_path, sep='\t', header=None, nrows=leave_num)
        df.columns = ['user', 'item', 'rating', 'time']

        # 按用户和时间排序
        df = df.sort_values(['user', 'time'])

        # 移除评分小于阈值的交互
        df = df[df['rating'] > rating_threshold]

        user_item = df[['user', 'item']].values.tolist()
        user_item = [list(map(str, line)) for line in user_item]

        users, items = zip(*user_item)
        user_list = []
        item_list = []

        # 统计每个用户出现的次数
        users_cnt = Counter(users)
        for user, count in users_cnt.items():
            if count >= user_threshold:
                user_list.append(user)

        # 统计每个物品出现的次数
        items_cnt = Counter(items)
        for item, count in items_cnt.items():
            if count >= item_threshold:
                item_list.append(item)

        num_user, num_item = len(user_list), len(item_list)

        # 将用户和物品映射到 ID
        user2id, item2id = map_elements_to_id(user_list), map_elements_to_id(item_list)
        self.name_id['item'] = item2id
        return (
            np.array(remove_unrating_user_and_rename(user2id, item2id, user_item)),
            num_user,
            num_item,
            user2id,
            item2id
        )

    def get_item_entity_pairs(self, subdir):
        """
        获取物品-其他实体对

        Args:
            subdir (`str`): 子目录

        Returns:
            b_other (`list`): 物品-其他实体对
            others2id (`dict`): 其他实体 ID 映射
            len(others) (`int`): 其他实体数量
        """
        b_other = []
        file_path = self.dir + subdir
        print(file_path)
        with codecs.open(file_path, 'r', encoding=self.encoding) as rfile:  # iso-8859-15
            for line in rfile:
#                if self.name in ['ml100k']:
#                    line = line.strip().split(',')
#                if self.name in ['ML','tiktok' ,'CiaoDVD','Games','Amazon_App','dianping','Grocery','Kindle']:
                line = line.strip().split('\t')
                if len(line) != 2:
                    print(line)
                b_other.append([str(line[0]), str(line[1])])
        bs, others = zip(*b_other)
        others = list(set(others))
        others2id = map_elements_to_id(others)
        self.name_id[subdir] = others2id
        return remove_unrating_user_and_rename(self.dict_entity2id['item'],others2id,b_other), others2id,len(others)    

    def get_u_other(self,subdir):
        b_other = []
        file_path = self.dir + subdir
        with open(file_path) as rfile:
            for line in rfile:
                if self.name=='yelp50k' or self.name=='yelp200k' or self.name=='doubanDVD':
                    line = line.strip().split()
                if self.name=='amazon':
                    line = line.strip().split(',')
                if self.name=='ml100k':
                    line = line.strip().split(',')
                b_other.append(line)
        bs,others = zip(*b_other)
        others = list(set(others))
        others2id = map_elements_to_id(others)
        return remove_unrating_user_and_rename(self.dict_entity2id['user'],others2id,b_other) , others2id,len(others)    

    def constrcut_markov_matrices(self, keep=1):
        """
        构建马尔可夫矩阵

        Args:
            keep (`int`): 保留的交互次数

        Returns:
            markov (`lil_matrix`): 马尔可夫矩阵
        """
        markov = lil_matrix((self.entity_num['item'], self.entity_num['item']))
        for user, item in tqdm(self.train_set,'markov_preparing'):
            item_ps = self.latest_interaction[(user, item)]
            for item_p in item_ps[-keep:]:
                if item_p >= 0:
                    markov[item_p,item] +=1
        return markov

    def pic_feature(self, name, dims=64):
        """
        获取物品的图片特征

        Args:
            name (`str`): 特征文件名
            dims (`int`): 特征维度

        Returns:
            pic_feature (`numpy.ndarray`): 物品的图片特征
        """
        pic_feature = np.zeros([self.entity_num['item'],dims])
        # read feature
        file_path = self.dir + name  # 'feature.npy'
        feature = np.load(file_path)
        # read items rank
        with codecs.open(self.dir + 'feature.data','r',encoding=self.encoding) as rfile:  # iso-8859-15
            for line in rfile:
                line = line.strip().split('\t')
                item = line[0]
                idx =int(line[1])
                if item in self.dict_entity2id['item']:
                    pic_feature[self.dict_entity2id['item'][item]] = feature[idx]
        return pic_feature

    def holdout_users(self,test,n):
#        us,bs = zip(*test)
#        C_us = Counter(us)
#        u_num =np.array([ list(C_us.keys()),list(C_us.values())],dtype=np.int)
        ret = []
        for ui in test:
            if ui[0] >n:
                return ret
            else:
                ret.append(ui)
        return ret
