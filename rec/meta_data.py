import copy
import numpy as np


class MetaData(object):
    def __init__(self, args, data):
        self.args = args
        self.data = data
        self.n_user = self.data.entity_num['user']
        self.n_item = self.data.entity_num['item']

        # (user, item) -> index
        self.user_item_pairs_to_index = {
            (line[0], line[1]): i
            for i, line in enumerate(data.dict_list['user_item'])
        }

        self.base_models = self.args['base_model']
        self.top_k_items = 100  # 这个top_k_items是为了取前多少个items的score，大于top_k_items的设为0.

        self.meta = []
        for base_model in self.base_models:
            res = np.load(self.args['path'] + f"/{base_model}.npy")
            self.meta.append(res)
        meta = np.stack(self.meta, axis=1)

        # meta_XXXX 表示 [user, item, ranking(102)]
        self.train_meta = meta[[
            self.user_item_pairs_to_index[line[0], line[1]]
            for line in self.data.valid_set
        ]]
        self.test_meta = meta[[
            self.user_item_pairs_to_index[line[0], line[1]]
            for line in self.data.test_set
        ]]

        # 返回对于 ensemble 的训练集和 basemodel 值
        self.user_item_pairs, self.user_item_pairs_labels = self.label_positive()

        # 初始化 users 属性
        self.users = None

    def all_score(self, traintest):
        """
        返回所有得分的函数

        Args:
            traintest (`np.ndarray`): 训练集或测试集

        Returns:
            u_k_i (`np.ndarray`): 所有得分
        """
        rank_chunk = traintest[:,:,2:2+self.top_k_items] #[batch,k,rank]
        btch, k, n = rank_chunk.shape  # [batch, k, rank]
        rank_chunk_reshape = np.reshape(rank_chunk, [-1, n])

        u_k_i = np.zeros([btch*k, self.n_item])     #[batch,k,n_item]
        for i in range(n):
            u_k_i[np.arange(len(u_k_i)), rank_chunk_reshape[:,i]] = 1 / (i + 10)
        return np.reshape(u_k_i, [btch, k, self.n_item])

    def label_positive(self):
        """
        返回正样本得分的函数

        Returns:
            tuple: 包含用户-物品对和对应的得分
        """
        # 在需要时设置 users 值
        user_item_pairs = np.array(self.data.train_set)
        self.users = user_item_pairs[:, 0]

        # 获取基础模型的数量
        n_base_models = len(self.base_models)
        # 创建得分矩阵，形状为 [样本数, 基模型数]
        label = np.zeros([len(self.train_meta), n_base_models])
        # 获取真实 (Ground Truth) 物品 ID，扩展维度成 [batch, 1]
        gt_item = np.expand_dims(self.train_meta[:, 0, 1], axis=1)

        # 复制基模型预测的前 N 个排名结果，形状为 [batch, k, N]，k 是基模型数量，N 是考虑的排名数量
        rank_chunk = copy.deepcopy(self.train_meta[:, :, 2: 2+self.top_k_items])  # [batch, k, rank]

        for k in range(n_base_models):
            # 获取当前基模型的排名结果
            rank_chunk_k = rank_chunk[:, k, :]
            # 比较真实物品是否在排名中
            is_item_in_rank = gt_item == rank_chunk_k
            # 如果物品在排名中，计算得分 = 1 / (10 + 物品的排名位置)
            label[np.sum(is_item_in_rank, axis=1) > 0, k] = 1 / (10 + np.argwhere(is_item_in_rank)[:, 1])

        # 返回用户-物品对和对应的得分
        return self.train_meta[:, 0, :2], label

    def label_negative(self, neglist, NG):  # neglist 是一个一维列表，其中每个元素表示负样本物品
        """
        返回负样本得分的函数

        Args:
            neglist (`np.ndarray`): 负样本列表
            NG (`int`): 负样本数量

        Returns:
            label (`np.ndarray`): 负样本得分
        """
        n_k = len(self.base_models)
        assert len(neglist) == len(self.train_meta), 'wrong size'

        label = []
        for i in range(NG):
            label_i = np.zeros([len(self.train_meta), n_k])
            GT_item = np.expand_dims(neglist[:,i],axis=1)#[batch,1] #for item
            rank_chunk = copy.deepcopy(self.train_meta[:,:,2:2+self.top_k_items]) #[batch,k,rank]       
            for k in range(n_k):
                rank_chunk_k = rank_chunk[:, k, :]
                torf = GT_item == rank_chunk_k
                label_i[np.sum(torf,axis=1)>0,k] = 1 / (10+ np.argwhere(torf)[:,1])
            label.append(label_i)
        return np.stack(label, axis=1)
