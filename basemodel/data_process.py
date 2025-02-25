import copy
import numpy as np
import pandas as pd
from collections import Counter
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix, save_npz
import codecs
from tqdm import tqdm

leave_num = 1000000000
remove_rating = 2
last = 5  # 5 for BMS; 10 for SNR


def relation_dict(n1, n2, list1):
    # 将总数n1，n2的entity转化成映射字典
    dict_forward = {i:[] for i in range(n1)}
    dict_reverse = {i:[] for i in range(n2)}
    for x, y in list1:
        if y not in dict_forward[x]:
            dict_forward[int(x)].append(int(y))
            dict_reverse[int(y)].append(int(x))
    return dict_forward


def remove_unrating_user_and_rename(tranA, tranB, list1):
    # 去掉没有的user对应的rating，并rename
    res = []
    for x, y in list1:
        if x in tranA.keys() and y in tranB.keys():
            res.append([tranA[x], tranB[y]])
    return res


def reverse_and_map(l):
    # 做rename，???->n
    return {v:k for k, v in enumerate(l)}


def set_forward(dict1):
    return_1 = dict()
    for key in dict1.keys():
        sub = dict1[key]
        return_1[key] = {ky:set(sub[ky]) for ky in sub.keys()}
    return return_1


def build_sparse_matrix(n_1, n_2, list1):
    res = lil_matrix((n_1, n_2))
    for user in list1.keys():
        for item in list1[user]:
            res[user, item] = 1
    return res


class Data(object):
    """
    数据处理类，用于处理和准备图神经网络所需的数据。

    属性:
        name_id (`dict`): 实体名称到ID的映射字典
        name (`str`): 数据集名称
        dir (`str`): 数据文件路径
        encoding (`str`): 文件编码格式
        entity (`list`): 实体类型列表，如['user','item','G']
        user_side_entity (`list`): 用户侧实体类型列表
        item_side_entity (`list`): 物品侧实体类型列表
        drop_user (`int`): 用户交互数阈值，少于此值的用户将被过滤
        drop_item (`int`): 物品交互数阈值，少于此值的物品将被过滤
        dict_entity2id (`dict`): 各实体到ID的映射字典
        dict_list (`dict`): 实体间关系的二元组列表字典
        dict_forward (`dict`): 实体间正向关系字典
        dict_reverse (`dict`): 实体间反向关系字典
        entity_num (`dict`): 各类实体的数量统计
    """
    def __init__(self, args, seed=0, Markov=False):
        self.name_id = dict()
        self.name = args.dataset
        self.dir = args.path if args.path[-1] == '/' else args.path + '/'

        if args.name in ['CiaoDVD', 'dianping']:
            self.encoding = 'utf-8'
        else:
            self.encoding = 'iso-8859-15'

        # 0: 列名称, 1: 用户侧实体, 2: 物品侧实体, 3: 用户过滤阈值, 4: 物品过滤阈值
        entity_config = {
            'Amazon_App': (['user', 'item', 'G'], [], ['G'], 10, 10),
            'Games': (['user', 'item', 'G'], [], ['G'], 10, 10),
            'ml1m': (['user', 'item', 'G'], [], ['G'], 10, 10),
            'Grocery': (['user', 'item', 'G'], [], ['G'], 20, 20),
            'Sport': (['user', 'item', 'G'], [], ['G'], 5, 5),
            'Clothing': (['user', 'item', 'G'], [], ['G'], 5, 5),
            'Instant_Video': (['user', 'item', 'G'], [], ['G'], 5, 5),
            'Pet_Supplies': (['user', 'item', 'G'], [], ['G'], 5, 5),
            'Patio': (['user', 'item', 'G'], [], ['G'], 5, 5),
            'Office_Products': (['user', 'item', 'G'], [], ['G'], 5, 5),
            'Musical_Instruments': (['user', 'item', 'G'], [], ['G'], 5, 5),
            'Digital_Music': (['user', 'item', 'G'], [], ['G'], 5, 5),
            'Baby': (['user', 'item', 'G'], [], ['G'], 5, 5),
            'Automotive': (['user', 'item', 'G'], [], ['G'], 5, 5),
            'dianping': (['user', 'item', 'S', 'A', 'C'], [], ['S', 'A', 'C'], 20, 20),
            'Kindle': (['user', 'item', 'G'], [], ['G'], 50, 50),
            'tiktok': (['user', 'item', 'G'], [], ['G'], 50, 50),
            'ml-25m': (['user', 'item', 'G'], [], ['G'], 10, 10, 'csv'),
            'ml-10M100K': (['user', 'item', 'G'], [], ['G'], 10, 10, 'dat')
        }

        if self.name in entity_config:
            self.entity, self.user_side_entity, self.item_side_entity = entity_config[self.name][:3]
            self.drop_user, self.drop_item = entity_config[self.name][3:5]
            self.data_suffix = entity_config[self.name][5] if len(entity_config[self.name]) > 5 else None

        self.dict_entity2id = {}  # entity对应的ID
        self.dict_list = {}  # entity1_entity2对应的两元组
        self.dict_forward = {}  # dict entity1_entity2对应的关系字典
        self.dict_reverse = {}
        self.entity_num = {}  # 数量

        # 读取评分并划分训练集和测试集
        # 移除交互次数少于5的用户和物品
        self.dict_list['user_item'], self.entity_num['user'], self.entity_num['item'],\
        self.dict_entity2id['user'], self.dict_entity2id['item'] = self._load_rating(remove_rating, self.drop_user, self.drop_item, self.data_suffix)
        self.latest_interaction = self.find_latest_interaction(self.dict_list['user_item'])  # 时间特征

        self.train_set, self.valid_set, self.test_set = self.split_traintest(self.dict_list['user_item'])

        # 时间序列特征
        self.dict_forward['train'] = relation_dict(
            self.entity_num['user'],
            self.entity_num['item'],
            self.train_set
        )
        self.dict_forward['valid'] = relation_dict(
            self.entity_num['user'],
            self.entity_num['item'],
            self.valid_set
        )
        self.dict_forward['test'] = relation_dict(
            self.entity_num['user'],
            self.entity_num['item'],
            self.test_set
        )

        self.markov = self.constrcut_markov_matrices()

        # 将物品侧实体转换为其他实体并设置它们的正向和反向字典
        for entity in self.item_side_entity:
            # item_data_name = 'I' + entity + '.data'
            item_data_name = 'movies.csv'

            self.dict_list['item_' + entity], \
            self.dict_entity2id[entity], \
            self.entity_num[entity] = self._load_item_side_entities(item_data_name)

            # 构建物品侧实体的正向关系字典
            self.dict_forward['item_' + entity] = relation_dict(
                self.entity_num['item'],
                self.entity_num[entity],
                self.dict_list['item_' + entity]
            )

        # 读取用户侧实体
        for entity in self.user_side_entity:
            self.dict_list['user_' + entity], \
            self.dict_entity2id[entity], \
            self.entity_num[entity] = self._load_item_side_entities('U' + entity + '.data')

            # 构建用户侧实体的正向关系字典
            self.dict_forward['user_' + entity] = relation_dict(
                self.entity_num['user'],
                self.entity_num[entity],
                self.dict_list['user_' + entity]
            )

#       build sparse matrice of entity-entity
        #user -item
        self.matrix = dict()
        self.matrix['user_item'] = build_sparse_matrix(self.entity_num['user'],self.entity_num['item'],self.dict_forward['train'])
        self.matrix['item_user'] = self.matrix['user_item'].transpose()

        #user - ?
        for entity in self.user_side_entity:
            self.matrix['user' + entity] = build_sparse_matrix(self.entity_num['user'],self.entity_num[entity],self.dict_forward['user_'+entity])
            self.matrix[entity + 'user'] = self.matrix['user'+entity].transpose()
        #item - ?
        for entity in self.item_side_entity:
            self.matrix['item' + entity] = build_sparse_matrix(self.entity_num['item'],self.entity_num[entity],self.dict_forward['item_'+entity])
            self.matrix[entity + 'item'] = self.matrix['item'+entity].transpose()

        self.set_forward = set_forward(self.dict_forward)
        print('user:%d\t item:%d\t train:%d\t'%(self.entity_num['user'], self.entity_num['item'], len(self.train_set)))

        if self.name in ['Grocery', 'Kindle', 'Games']:
            self.pic = self.pic_feature('feature.npy')
        if self.name in ['tiktok']:
            self.pic = self.pic_feature('visual.npy')
            self.acou = self.pic_feature('acoustic.npy')

    def split_traintest(self, u_i, ratio=[0.6, 0.8]):
        # 保持6:2:2和JMLR中的一致
        user_count = Counter(u_i[:,0])  # count number of user's interaction
        train = []
        valid = []
        test = []
        n = 0
        for i, user_item in enumerate(u_i):
            user, item = user_item
            if n < min(int(user_count[user] * ratio[0]), user_count[user] - 4):
                train.append([user, item])
                n = n + 1
            # leave one out:user_count[user]-1:
            # leave percent out: min(int(user_count[user]*ratio[1]),user_count[user]-1):
            elif n < user_count[user]-1:
                valid.append([user, item])
                n = n + 1
            else:
                test.append([user, item])
            try:
                if u_i[i + 1][0] != user:
                    n = 0
            except:
                pass
        return train, valid, test

    def find_latest_interaction(self, u_i, keep=last):
        """
        找到用户最后一次交互的时间
        """
        result = dict()
        init = [-1 for i in range(keep)]
        latest = copy.deepcopy(init)
        for i, user_item in tqdm(enumerate(u_i), total=len(u_i), desc="Processing user-item interactions"):
            user, item = user_item
            result[(user, item)] = copy.deepcopy(list(latest[-keep:]))
            if i < len(u_i) - 1:
                if u_i[i + 1][0] != user:
                    latest = copy.deepcopy(init)
                else:
                    latest.append(item)

        return result

    def _load_rating(self, score, remove_less_than1, remove_less_than2, data_suffix):
        """
        获取评分数据

        Args:
            score (`float`): 评分阈值
            remove_less_than1 (`int`): 用户交互数阈值
            remove_less_than2 (`int`): 物品交互数阈值
            data_suffix (`str`): 数据文件后缀

        Returns:
            `tuple`: 返回用户物品交互列表、用户数量、物品数量、用户ID映射字典、物品ID映射字典
        """
        if data_suffix == 'csv':
            df = pd.read_csv(self.dir + 'ratings.csv')
        elif data_suffix == 'dat':
            df = pd.read_csv(self.dir + 'ratings.dat', sep='::', names=['userId', 'movieId', 'rating', 'timestamp'])
        else:
            df = pd.read_csv(self.dir + 'ratings.data', sep='\t', header=None, nrows=leave_num)
        df.columns = ['user','item','rating','time']
        df = df.sort_values(['user','time'])  # sort for user and time

        # 删除评分小于 score 的交互
        df = df[df['rating'] > score]

        # 获取用户物品交互列表
        u_i = df[['user','item']].values.tolist()
        u_i = [list(map(str,line)) for line in u_i]

        us,bs = zip(*u_i)
        us_list = []
        bs_list = []
        C_us = Counter(us)
        for u in C_us.keys():
            if C_us[u]>=remove_less_than1:
                us_list.append(u)
        C_bs = Counter(bs)
        for b in C_bs.keys():
            if C_bs[b]>=remove_less_than2:
                bs_list.append(b)
        num_user, num_item = len(us_list), len(bs_list)
        user2id, item2id = reverse_and_map(us_list), reverse_and_map(bs_list)
        self.name_id['item'] = item2id

        return np.array(remove_unrating_user_and_rename(user2id,item2id, u_i)),num_user, num_item,user2id, item2id

    def _load_item_side_entities(self, subdir):
        """
        获取物品侧实体

        Args:
            subdir (`str`): 子目录名称

        Returns:
            `tuple`: 返回物品侧实体列表、物品侧实体ID映射字典、物品侧实体数量
        """
        item_entities = []
        file_path = self.dir + subdir
        print(file_path)

        if file_path.endswith('.csv'):
            items = pd.read_csv(file_path)
            item_entities = items.iloc[:, :2].values.tolist()  # 只取前两列
            item_entities = [[str(x), str(y)] for x, y in item_entities]
        elif file_path.endswith('.dat'):
            items = pd.read_csv(file_path, sep='::', names=['movieId', 'title', 'genres'])
            item_entities = items.iloc[:, :2].values.tolist()  # 只取前两列
            item_entities = [[str(x), str(y)] for x, y in item_entities]
        else:
            with codecs.open(file_path, 'r', encoding=self.encoding) as rfile:  # iso-8859-15
                for line in rfile:
                    line = line.strip().split('\t')
                    if len(line) != 2:
                        print(line)
                    item_entities.append([str(line[0]), str(line[1])])

        item_ids, other_entities = zip(*item_entities)
        unique_other_entities = list(set(other_entities))

        other_entities2id = reverse_and_map(unique_other_entities)
        self.name_id[subdir] = other_entities2id

        return remove_unrating_user_and_rename(
            self.dict_entity2id['item'],
            other_entities2id,
            item_entities
        ), other_entities2id, len(unique_other_entities)

    def get_u_other(self, subdir):
        """
        获取用户侧实体

        Args:
            subdir (str): 子目录名称

        Returns:
            tuple: 返回用户侧实体列表、用户侧实体ID映射字典、用户侧实体数量
        """
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
        others2id = reverse_and_map(others)
        return remove_unrating_user_and_rename(self.dict_entity2id['user'],others2id,b_other) , others2id,len(others)    
    
    def constrcut_markov_matrices(self, keep=1):
        markov = lil_matrix((self.entity_num['item'], self.entity_num['item']))

        for user, item in tqdm(self.train_set, 'markov_preparing'):
             item_ps = self.latest_interaction[(user, item)]
             for item_p in item_ps[-keep:]:
                 if item_p >= 0:
                     markov[item_p, item] += 1
        return markov
    
    def pic_feature(self,name,dims=64):
        pic_feature = np.zeros([self.entity_num['item'],dims])
        #read feature
        file_path = self.dir + name#'feature.npy'
        feature = np.load(file_path)
        #read items rank
        with codecs.open(self.dir + 'feature.data','r',encoding=self.encoding) as rfile:#iso-8859-15
            for line in rfile:
                line = line.strip().split('\t') 
                item = line[0]
                idx =int(line[1])
                if item in self.dict_entity2id['item']:
                    pic_feature[self.dict_entity2id['item'][item]] = feature[idx]
        return pic_feature
        
#                if self.name in ['ml100k']:
#                    line = line.strip().split(',')   
#                if self.name in ['ML', 'CiaoDVD','Amazon_App','dianping']:
#                    line = line.strip().split('\t')  
#                if (len(line)!=2):
#                    print(line)
#                b_other.append([str(line[0]),str(line[1])])
        
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
