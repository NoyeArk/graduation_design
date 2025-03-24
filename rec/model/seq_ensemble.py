import copy
import toolz
import argparse
import numpy as np
from tqdm import tqdm

import tensorflow as tf
variable_scope = tf.compat.v1.variable_scope
get_variable = tf.compat.v1.get_variable
placeholder = tf.compat.v1.placeholder
AUTO_REUSE = tf.compat.v1.AUTO_REUSE

from data_utils import Data
from module.utils import *

# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')

N = 100  # 这个N是为了取前多少个items的score，大于N的设为0.

# 基模型
base_models = ['acf', 'fdsa', 'harnn', 'caser', 'pfmc', 'sasrec', 'anam']

# 'ACF', 'FDSA', 'HARNN' 需要 attribute 信息，Caser', 'PFMC', 'SASRec' 仅依赖于序列信息。
print_train = False  # 是否输出 train 上的验证结果（过拟合解释）。


def parse_args(name, factor, batch_size, tradeoff, user_module, model_module, div_module, epoch, maxlen):
    parser = argparse.ArgumentParser(description="Run .")
    parser.add_argument('--name', nargs='?', default= name)
    parser.add_argument('--model', nargs='?', default='SASEM')
    parser.add_argument('--path', nargs='?', default='D:/Code/graduation_design/data/'+name,
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default=name,
                        help='Choose a dataset.')
    parser.add_argument('--batch_size', type=int, default=batch_size,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=factor,
                        help='Number of hidden factors.')
    parser.add_argument('--lamda', type=float, default = 0.00001,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--epoch', type=int, default=epoch)
    parser.add_argument('--tradeoff', type=float, default=tradeoff)
    parser.add_argument('--user_module', nargs='?', default=user_module)
    parser.add_argument('--model_module', nargs='?', default=model_module)
    parser.add_argument('--div_module', nargs='?', default=div_module)
    parser.add_argument('--maxlen', type=int, default=maxlen)
    parser.add_argument('--optimizer', nargs='?', default='AdamOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    return parser.parse_args()


class Model(object):
    """
    模型类
    """
    def __init__(self, args, data, max_len):
        """
        初始化模型

        Args:
            args (`argparse.Namespace`): 参数
            data (`Data`): 数据
            hidden_factor (`int`): 隐藏因子
            learning_rate (`float`): 学习率
            lamda_bilinear (`float`): 正则化系数
            optimizer_type (`str`): 优化器类型
        """
        self.args = args
        self.data = data
        self.n_user = self.data.entity_num['user']
        self.n_item = self.data.entity_num['item']
        self.learning_rate = args['lr']
        self.hidden_factor = args['hidden_dim']
        self.lamda_bilinear = args['lamda']
        self.optimizer_type = args['optimizer']
        self.n_base_model = len(base_models)
        self.seq_max_len = max_len

        # 初始化 prediction 属性
        self.prediction = None

        self._init_graph()

    def _init_graph(self):
        """
        初始化 tensorflow 图。
        """
        self.graph = tf.compat.v1.Graph()

        with self.graph.as_default():
            self.weights = self._initialize_weights()

            # 将 tf.placeholder 替换为 tf.compat.v1.placeholder
            self.user_id = tf.compat.v1.placeholder(tf.int32, shape=[None])
            self.item_id_pos = tf.compat.v1.placeholder(tf.int32, shape=[None])
            self.item_id_neg = tf.compat.v1.placeholder(tf.int32, shape=[None,None])
            self.input_seq = tf.compat.v1.placeholder(tf.int32, shape=[None, self.seq_max_len])
            self.meta_pos = tf.compat.v1.placeholder(tf.float32, shape=[None, self.n_base_model])
            self.meta_neg = tf.compat.v1.placeholder(tf.float32, shape=[None, None, self.n_base_model])
            self.base_model_focus = tf.compat.v1.placeholder(tf.int32, shape=[None, self.n_base_model, self.seq_max_len])
            self.times = tf.compat.v1.placeholder(tf.float32, shape=[None])
            self.is_training = tf.compat.v1.placeholder(tf.bool, shape=())
            self.meta_all_items = tf.compat.v1.placeholder(tf.float32, shape=[None, self.n_base_model, self.n_item])

            # 用户嵌入 [none, d]
            self.user_embed = tf.nn.embedding_lookup(self.weights['user_embeddings'], self.user_id)
            # 条件
            self.condition = tf.cast(
                tf.greater(tf.reduce_sum(self.meta_pos, axis=1), 0),
                dtype=tf.float32
            )

            # 对用户进行注意力建模
            if self.args['user_module'] == 'MC':
                self.item_sequence_embs = tf.nn.embedding_lookup(
                    self.weights['item_embeddings1'],
                    self.input_seq
                )  # [none, p, d]
                self.preference = self.user_embed + tf.reduce_sum(self.item_sequence_embs,axis=1)
                self.items_embs = self.weights['item_embeddings1']
            elif self.args['user_module'] == 'GRU':
                self.item_sequence_embs = tf.nn.embedding_lookup(
                    self.weights['item_embeddings1'],
                    self.input_seq
                )  # [none,p,d]
                lstm_cell = tf.contrib.rnn.GRUCell(self.hidden_factor)
                value, preference = tf.nn.dynamic_rnn(
                    lstm_cell,
                    self.item_sequence_embs,
                    dtype=tf.float32
                )  # [none, 5, d]
                self.preference = value[:,-1,:] + self.user_embed #[none,d]
                self.items_embs = self.weights['item_embeddings1']
            elif self.args['user_module'] == 'LSTM':
                self.item_sequence_embs = tf.nn.embedding_lookup(
                    self.weights['item_embeddings1'],
                    self.input_seq
                )  # [none,p,d]
                lstm_cell = tf.contrib.rnn.LSTMCell(self.hidden_factor)
                value, preference = tf.nn.dynamic_rnn(
                    lstm_cell,
                    self.item_sequence_embs,
                    dtype=tf.float32
                )  # [none, 5, d]
                self.preference = value[:, -1, :] + self.user_embed  # [none,d]
                self.items_embs = self.weights['item_embeddings1']
            elif self.args['user_module'] == 'SAtt':
                self.state = self.FFN(self.input_seq)
                self.preference = self.state[:, -1, :] + self.user_embed  # [none,d]
                self.items_embs = self.item_emb_table
            elif self.args['user_module'] == 'static':
                self.preference = self.user_embed
                self.items_embs = self.weights['item_embeddings1']

            # 对基模型进行注意力建模
            if self.args['model_module'] == 'dynamic':
                self.each_model_emb = tf.nn.embedding_lookup(
                    self.items_embs,
                    self.base_model_focus
                )  # [none, k, p, d]

                # 计算每个时间步的权重
                self.wgt_model = tf.reshape(
                    tensor=tf.constant(
                        1 / np.log2(np.arange(self.seq_max_len) + 2),
                        dtype=tf.float32
                    ),
                    shape=[1, 1, -1, 1]
                )

                # 计算每个基模型的嵌入
                self.basemodel_emb = self.weights['base_model_embeddings'] + tf.reduce_sum(
                    self.wgt_model * self.each_model_emb,
                    axis=2
                )  # [none, k, d]
            elif self.args['model_module'] == 'static':
                self.basemodel_emb = self.weights['base_model_embeddings']  # [1, k, d]

            # 计算每个基模型的权重
            self.wgts_org = tf.reduce_sum(
                tf.expand_dims(self.preference, axis=1) * self.basemodel_emb,
                axis=-1
            )  # [none, n_k]
            self.wgts = tf.nn.softmax(self.wgts_org, axis=-1)  # [none, n_k]

            # 计算正负样本损失
            self.score_positive = tf.reduce_sum(self.meta_pos * self.wgts, axis=1)#none
            self.score_negative = tf.reduce_sum(
                self.meta_neg * tf.expand_dims(self.wgts, axis=1),
                axis=-1
            )  # none * NG
            self.loss_rec = self.pairwise_loss(self.score_positive, self.score_negative)

            # 计算正则化损失
            self.loss_reg = 0
            for wgt in tf.compat.v1.trainable_variables():
                self.loss_reg += self.lamda_bilinear * tf.nn.l2_loss(wgt)

            # 计算多样性损失
            if self.args['div_module'] == 'AEM-cov':
                # AEM diversity
                self.model_emb = self.basemodel_emb#[none,k,p]
                cov_idx = tf.constant(
                    1 - np.expand_dims(np.diag(np.ones(self.n_base_model)), axis=0),
                    dtype=tf.float32
                )  # [none, k, k]
                cov_div1 = tf.square(
                    tf.reduce_sum(
                        tf.expand_dims(self.model_emb, axis=1) * tf.expand_dims(self.model_emb,axis=2),
                        axis=-1
                    )
                )
                l2 = tf.reduce_sum(self.model_emb ** 2, axis=-1)  # none * k
                cov_div2 = tf.matmul(tf.expand_dims(l2, axis=-1), tf.expand_dims(l2, axis=1))  # none* k *k
                self.cov = cov_div1 / cov_div2  # [none, k, k]
            elif self.args['div_module'] == 'cov':
                self.model_emb =  self.basemodel_emb#[none,k,p]
                cov_wgt = tf.stop_gradient(
                    tf.expand_dims(self.wgts, axis=1) + tf.expand_dims(self.wgts ,axis=2)
                )  # [none, k, k]
                cov_idx = tf.constant(1-np.expand_dims(np.diag(np.ones(self.n_base_model)),axis=0),dtype=tf.float32)#[none,k,k]    
                cov_div = tf.square(tf.reduce_sum(tf.expand_dims(self.model_emb,axis=1)*tf.expand_dims(self.model_emb,axis=2),axis=-1))
                coff = cov_wgt * tf.reshape(self.times,[-1,1,1])
                self.cov = cov_idx * (1 - cov_div)  # [none, k, k]
                self.cov = self.cov * coff

            # 计算多样性损失
            self.loss_diversity = -self.args['tradeoff'] * tf.reduce_sum(self.cov)
            self.loss = self.loss_rec + self.loss_reg + self.loss_diversity

            # 选择优化器
            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer = tf.compat.v1.train.AdamOptimizer(
                    learning_rate=self.learning_rate,
                    beta1=0.9,
                    beta2=0.999,
                    epsilon=1e-8
                ).minimize(self.loss)
            elif self.optimizer_type == 'AdagradOptimizer':
                self.optimizer = tf.compat.v1.train.AdagradOptimizer(
                    learning_rate=self.learning_rate
                ).minimize(self.loss)
            elif self.optimizer_type == 'SGD':
                self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(
                    learning_rate=self.learning_rate
                ).minimize(self.loss)

            # 计算所有物品的得分
            out = tf.reduce_sum(
                tf.expand_dims(self.wgts, axis=2) * self.meta_all_items,
                axis=1
            )  # [none, n_item]
            self.out_all_topk = tf.nn.top_k(out, 200)

            # 初始化 saver
            self.saver = tf.compat.v1.train.Saver(max_to_keep=1)

            # 初始化会话
            self.sess = self._init_session()

            # 初始化所有变量
            init = tf.compat.v1.global_variables_initializer()
            self.sess.run(init)

    def FFN(self, input_seq):
        """
        前馈神经网络
        """
        mask = tf.expand_dims(
            tf.cast(tf.not_equal(input_seq, -1), dtype=tf.float32),
            -1
        )
        
        with tf.compat.v1.variable_scope("FFN", reuse=tf.compat.v1.AUTO_REUSE):
            # sequence embedding, item embedding table
            self.seq, item_emb_table = embedding(input_seq,
                                                 vocab_size=self.n_item + 1, #error?
                                                 num_units=self.hidden_factor,
                                                 zero_pad=True,
                                                 scale=True,
                                                 l2_reg=self.lamda_bilinear,
                                                 scope="input_embeddings",
                                                 with_t=True,
                                                 reuse=tf.compat.v1.AUTO_REUSE
                                                 )
            self.item_emb_table = item_emb_table[1:]

            # Positional Encoding
            t, pos_emb_table = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
                vocab_size=5,
                num_units=self.hidden_factor,
                zero_pad=False,
                scale=False,
                l2_reg=self.lamda_bilinear,
                scope="dec_pos",
                reuse=tf.compat.v1.AUTO_REUSE,
                with_t=True
            )
            self.seq += t

            # Dropout
            dropout_layer = tf.keras.layers.Dropout(rate=0.5)
            self.seq = dropout_layer(
                self.seq,
                training=tf.convert_to_tensor(self.is_training)
            )
            self.seq *= mask

            # Build blocks
            for i in range(2):
                with tf.compat.v1.variable_scope("num_blocks_%d" % i):
                    # 自注意力
                    self.seq = multihead_attention(queries=normalize(self.seq),
                                                   keys=self.seq,
                                                   num_units=self.hidden_factor,
                                                   num_heads=2,
                                                   dropout_rate=0.5,
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   scope="self_attention")

                    # Feed forward
                    self.seq = feedforward(normalize(self.seq), num_units=[self.hidden_factor,self.hidden_factor],
                                           dropout_rate=0.5, is_training=self.is_training)
                    self.seq *= mask

            self.seq = normalize(self.seq)
        seq_emb = tf.reshape(self.seq, [tf.shape(input_seq)[0] , self.seq_max_len, self.hidden_factor])
        return seq_emb

    def _init_session(self):
        """
        初始化会话。

        Returns:
            session (`tf.Session`): 会话
        """
        # adaptively growing video memory
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.allow_soft_placement = True
        return tf.compat.v1.Session(config=config)

    def _initialize_weights(self):
        """
        初始化权重。

        Returns:
            all_weights (`dict`): 所有权重
        """
        all_weights = dict()
        all_weights['user_embeddings'] = tf.Variable(
            np.random.normal(0.0, 0.01, [self.n_user, self.hidden_factor]),
            dtype=tf.float32
        )  # features_M * K
        all_weights['item_embeddings1'] = tf.Variable(
            np.random.normal(0.0, 0.01, [self.n_item, self.hidden_factor]),
            dtype=tf.float32
        )  # features_M * K
        all_weights['item_embeddings2'] = tf.Variable(
            np.random.normal(0.0, 0.01, [self.n_item, self.hidden_factor]),
            dtype=tf.float32
        )  # features_M * K
        all_weights['base_model_embeddings'] = tf.Variable(
            np.random.normal(0.0, 0.01, [1, self.n_base_model, self.hidden_factor]),
            dtype=tf.float32
        )  # features_M * K
        all_weights['wgts'] = tf.nn.softmax(
            tf.Variable(np.random.normal(0.0, 0.01, [1, 1, self.seq_max_len, 1]), dtype=tf.float32),
            axis=2
        )  # features_M * K

        return all_weights

    def partial_fit(self, data):
        """
        拟合一个批次。

        Args:
            data (`dict`): 数据字典

        Returns:
            loss_rec (`tf.Tensor`): 正样本损失
            loss_diversity (`tf.Tensor`): 多样性损失
        """
        feed_dict = {
            self.user_id: data['u'],
            self.input_seq: data['seq'],
            self.item_id_pos: data['i_pos'],
            self.item_id_neg: data['i_neg'],
            self.times: data['times'],
            self.meta_pos: data['meta_pos'],
            self.meta_neg: data['meta_neg'],
            self.base_model_focus: data['base_focus'],
            self.is_training: True
        }

        loss_rec, loss_diversity, _ = self.sess.run(
            (self.loss_rec, self.loss_diversity, self.optimizer),
            feed_dict=feed_dict
        )
        return loss_rec, loss_diversity

    def pairwise_loss(self, postive, negative):
        """
        计算正负样本损失。

        Args:
            postive (`tf.Tensor`): 正样本得分
            negative (`tf.Tensor`): 负样本得分

        Returns:
            loss (`tf.Tensor`): 正负样本损失
        """
        return -tf.reduce_sum(tf.sigmoid((postive - negative)))

    def topk(self, user_item_pairs, last_interaction, items_score, base_focus):
        """
        计算 topk 得分。

        Args:
            user_item_pairs (`np.ndarray`): 用户-物品对
            last_interaction (`np.ndarray`): 最后一次交互
            items_score (`np.ndarray`): 物品得分
            base_focus (`np.ndarray`): 基模型表示

        Returns:
            pred_item (`np.ndarray`): 预测物品
            wgts (`np.ndarray`): 权重
        """
        feed_dict = {
            self.user_id: user_item_pairs[:, 0],
            self.input_seq: last_interaction,
            self.meta_all_items: items_score,
            self.base_model_focus: base_focus,
            self.is_training: False
        }

        _, pred_item = self.sess.run(self.out_all_topk, feed_dict)
        wgts = self.sess.run(self.wgts, feed_dict)
        return pred_item, wgts

    def save_model(self, save_path):
        """
        保存模型到指定路径
        
        Args:
            save_path (str): 保存模型的路径
        """
        self.saver.save(self.sess, save_path)
        print(f"Model saved to: {save_path}")

    def load_model(self, load_path):
        """
        从指定路径加载模型
        
        Args:
            load_path (str): 加载模型的路径
        """
        self.saver.restore(self.sess, load_path)
        print(f"Model loaded from: {load_path}")


class MetaData(object):
    """
    元数据类
    """
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

        # 加载基础模型预测结果
        self.meta = []
        for base_model in base_models:
            load = np.load(f"D:/Code/graduation_design/datasets/basemodel_v/{self.args.name}/{base_model}.npy")
            self.meta.append(load)
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
        rank_chunk = traintest[:,:,2:2+N] #[batch,k,rank]
        btch,k,n = rank_chunk.shape  # [batch, k, rank]
        rank_chunk_reshape = np.reshape(rank_chunk, [-1, n])

        u_k_i = np.zeros([btch*k,self.n_item])     #[batch,k,n_item]
        for i in range(n):
            u_k_i[np.arange(len(u_k_i)), rank_chunk_reshape[:,i]] = 1/(i+10)
        return np.reshape(u_k_i,[btch,k,self.n_item])

    def label_positive(self):
        """
        返回正样本得分的函数
        """
        # 在需要时设置 users 值
        user_item_pairs = np.array(self.data.train_set)
        self.users = user_item_pairs[:, 0]

        # 获取基础模型的数量
        n_base_models = len(base_models)
        # 创建得分矩阵，形状为 [样本数, 基模型数]
        label = np.zeros([len(self.train_meta), n_base_models])
        # 获取真实 (Ground Truth) 物品 ID，扩展维度成 [batch, 1]
        gt_item = np.expand_dims(self.train_meta[:, 0, 1], axis=1)

        # 复制基模型预测的前 N 个排名结果，形状为 [batch, k, N]，k 是基模型数量，N 是考虑的排名数量
        rank_chunk = copy.deepcopy(self.train_meta[:, :, 2: 2+N])  # [batch, k, rank]

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
        n_k = len(base_models)
        assert len(neglist) == len(self.train_meta), 'wrong size'

        label = []
        for i in range(NG):
            label_i = np.zeros([len(self.train_meta), n_k])
            GT_item = np.expand_dims(neglist[:,i],axis=1)#[batch,1] #for item
            rank_chunk = copy.deepcopy(self.train_meta[:,:,2:2+N]) #[batch,k,rank]       
            for k in range(n_k):
                rank_chunk_k = rank_chunk[:, k, :]
                torf = GT_item == rank_chunk_k
                label_i[np.sum(torf,axis=1)>0,k] = 1 / (10+ np.argwhere(torf)[:,1])
            label.append(label_i)
        return np.stack(label, axis=1)


class Train_MF(object):
    """
    训练类
    """
    def __init__(self, args, data, meta_data):
        self.args = args
        self.epoch = self.args.epoch
        self.batch_size = args.batch_size
        self.seq_max_len = args.maxlen
        # Data loading
        self.data = data
        self.meta_data = meta_data
        self.entity = self.data.entity
        self.n_user = self.data.entity_num['user']
        self.n_item = self.data.entity_num['item']
        # Training\\\建立模型
        self.model = Model(
            self.args,
            self.data,
            args.hidden_factor,
            args.lr,
            args.lamda,
            args.optimizer
        )
        with open("./process_result.txt", "a") as f:
             f.write("dataset:%s\n" % (args.name))

    def train(self):
        """
        训练函数，拟合数据
        """
        # 初始结果
        MAP_valid = 0
        p = 0

        max_map, max_ndcg, max_prec = (0, 0), (0, 0), (0, 0)
        for epoch in range(0, self.epoch + 1):  # 每一次迭代训练
            shuffle = np.arange(len(self.meta_data.user_item_pairs))
            # np.random.shuffle(shuffle)

            # 用户-物品对
            user_item_pairs = self.meta_data.user_item_pairs  # none * 2

            self.users = user_item_pairs[:, 0]
            self.times = self.timestamp()
            self.items = user_item_pairs[:, 1]

            # 采样，none * NG
            negative_sample_count = 1
            self.negative_samples = self.sample_negative(
                user_item_pairs=user_item_pairs,
                meta_data=self.meta_data.train_meta,
                negative_sample_count=negative_sample_count
            )

            # 序列 none * seq
            self.seq = np.array([self.data.latest_interaction[(line[0], line[1])] for line in user_item_pairs])

            # 正样本标签
            meta_positive = self.meta_data.user_item_pairs_labels #none * k  k denotes BM number

            # 负样本标签
            meta_negative = self.meta_data.label_negative(
                self.negative_samples,
                negative_sample_count
            )  # none * NG * k

            # 基模型训练
            base_focus = self.meta_data.train_meta[:, :, 2: 2 + self.seq_max_len] #none * k * p # p denotes window size

            # 批量训练
            for user_chunk in tqdm(toolz.partition_all(self.batch_size, [i for i in range(len(user_item_pairs))])):
                p = p + 1
                chunk = shuffle[list(user_chunk)]

                u_chunk = self.users[chunk]  # none
                seq_chunk = self.seq[chunk]  # none * p
                i_pos_chunk = self.items[chunk]  # none
                i_neg_chunk = self.negative_samples[chunk]  # none * NG

                # 正负样本标签
                meta_positive_chunk = meta_positive[chunk]  # none * k
                meta_negative_chunck = meta_negative[chunk]  # none * NG * k

                # 基模型表示
                base_focus_chunck = base_focus[chunk]
                times = self.times[chunk]

                self.feed_dict = {
                    'u': u_chunk,
                    'seq': seq_chunk,
                    'i_pos': i_pos_chunk,
                    'i_neg': i_neg_chunk,
                    'meta_pos': meta_positive_chunk,
                    'meta_neg': meta_negative_chunck,
                    'base_focus': base_focus_chunck,
                    'times': times
                }
                loss = self.model.partial_fit(self.feed_dict)

            # 评估训练和验证数据集
            if epoch % 1 == 0:
                print(f"Loss {loss[0]:.4f}\t{loss[1]:.4f}")

                # 评估训练和验证数据集
                if print_train:
                    init_test_TopK_train = self.evaluate_TopK(
                        test=self.data.valid_set[:10000],
                        test_meta=self.meta_data.train_meta[:10000],
                        topk=[10]
                    )
                    print(init_test_TopK_train)

                # 评估测试集
                init_test_TopK_test = self.evaluate_TopK(
                    test=self.data.test_set,
                    test_meta=self.meta_data.test_meta,
                    topk=[20, 50]
                )

                print(f"Epoch {epoch} \t TEST SET MAP: {init_test_TopK_test[0]:.4f}, NDCG: {init_test_TopK_test[1]:.4f}, PREC: {init_test_TopK_test[2]:.4f}\n")

                with open("./process_result.txt", "a") as f:
                    f.write(f"Epoch {epoch} \t TEST SET MAP: {init_test_TopK_test[0]:.4f}, NDCG: {init_test_TopK_test[1]:.4f}, PREC: {init_test_TopK_test[2]:.4f}\n")

                if MAP_valid < np.mean(init_test_TopK_test[4:]):
                    MAP_valid = np.mean(init_test_TopK_test[4:])
                    result_print = init_test_TopK_test

                max_map = (max(max_map[0], result_print[0]), max(max_map[1], result_print[3]))
                max_ndcg = (max(max_ndcg[0], result_print[1]), max(max_ndcg[1], result_print[4]))
                max_prec = (max(max_prec[0], result_print[2]), max(max_prec[1], result_print[5]))

        # 保存最终模型
        # self.model.save_model(f"./models/{self.args.name}_{self.args.model}/.ckpt")

        with open("./result.txt","a") as f:
            f.write("{},{},{},{},{},{},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f},{:.5f}\n".format(
                self.args.name,
                self.args.model,
                self.args.user_module,
                self.args.model_module,
                self.args.div_module,
                self.args.tradeoff,
                max_map[0],
                max_ndcg[0],
                max_prec[0],
                max_map[1],
                max_ndcg[1],
                max_prec[1]
                # result_print[0],
                # result_print[1],
                # result_print[2],
                # result_print[3],
                # result_print[4],
                # result_print[5]
            ))

    def sample_negative(self, user_item_pairs, meta_data, negative_sample_count):
        """
        采样负样本的函数

        Args:
            user_item_pairs (`np.ndarray`): 用户-物品对
            meta_data (`np.ndarray`): 元数据
            negative_sample_count (`int`): 负样本数量

        返回值:
            样本 (`np.ndarray`): 负样本
        """
        # 基础模型训练
        BM_train = self.data.dict_forward['train']

        # 只使用每个基础模型预测的前 50 个排名结果
        top_k_results = 50
        meta_data = meta_data[:, :, 2:2+top_k_results]

        # 重塑元数据
        meta_data = np.reshape(meta_data, [len(meta_data), -1])  # [none, k*50]
        num_rows, _ = meta_data.shape
        sample = []

        # 进行 negative_sample_count 次负样本生成
        for _ in range(negative_sample_count):
            # 随机生成样本
            sample_i = np.random.randint(0, self.n_item, num_rows)
            for j, item in enumerate(sample_i):
                # 如果样本在基础模型中，则重新生成样本
                if item in BM_train[user_item_pairs[j, 0]]:
                    sample_i[j] = np.random.randint(0, self.n_item)
            sample.append(sample_i)

        return np.stack(sample, axis=-1)

    def timestamp(self):
        """
        返回时间戳的函数
        """
        # 初始化时间戳
        t = np.ones(len(self.users))

        # 初始化计数器
        s, cout, c = self.users[0], 10, 1
        for i, ur in enumerate(self.users):
            if ur == s:
                cout += 1
                c += 1
                t[i] = cout
            else:
                t[i - c: i] = t[i - c: i] / cout
                cout = 10
                c = 1
                s = ur
                t[i] = cout
        t[i - c + 1:] = t[i - c + 1:] / cout
        return t

    def evaluate_TopK(self, test, test_meta, topk):
        """
        评估 TopK 的函数。

        Args:
            test (`np.ndarray`): 测试数据
            test_meta (`np.ndarray`): 测试元数据
            topk (`list`): TopK 列表

        Returns:
            result_MAP (`dict`): MAP 结果
            result_PREC (`dict`): PREC 结果
            result_NDCG (`dict`): NDCG 结果
        """
        # 获取测试数据
        user_item_pairs = copy.deepcopy(np.array(test))  # none * 2
        size = len(user_item_pairs)

        # 初始化结果字典
        result_map = {key: [] for key in topk}
        result_ndcg = {key: [] for key in topk}
        result_recall = {key: [] for key in topk}

        # 初始化每一个用户-物品对的最后一次交互
        last_iteraction = []  # none * 5
        for line in user_item_pairs:
            user, item = line
            last_iteraction.append(self.data.latest_interaction[(user, item)])

        # 将最后一次交互转换为数组
        last_iteraction = np.array(last_iteraction)

        # 分块处理
        num = 999  # self.n_user

        for i in range(int(size / num + 1)):
            beg, end = i * num, (i + 1) * num
            user_item_pairs_block = user_item_pairs[beg: end]
            last_iteraction_block = last_iteraction[beg: end]
            items_score = self.meta_data.all_score(test_meta[beg: end])
            base_focus = test_meta[beg:end, :, 2:2 + self.seq_max_len]

            # 预测得分
            pred_items, wgts = self.model.topk(
                user_item_pairs=user_item_pairs_block,
                last_interaction=last_iteraction_block,
                items_score=items_score,
                base_focus=base_focus
            )  # none * 50
            assert len(pred_items) == len(user_item_pairs_block)

            # 评估
            for i, (user, gt_item) in enumerate(user_item_pairs_block):
                # 对于每一个用户-物品对，计算 topk 的指标
                for top_n in topk:
                    useful_item_cnt = 0
                    for pred_item in pred_items[i]:
                        if useful_item_cnt == top_n:
                            result_map[top_n].append(0.0)
                            result_ndcg[top_n].append(0.0)
                            result_recall[top_n].append(0.0)
                            useful_item_cnt = 0
                            break
                        elif pred_item == gt_item:
                            result_recall[top_n].append(1.0)
                            result_ndcg[top_n].append(np.log(2) / np.log(useful_item_cnt + 2))
                            result_map[top_n].append(1 / (useful_item_cnt + 1))
                            useful_item_cnt = 0
                            break
                        elif pred_item in (self.data.set_forward['train'][user] or self.data.set_forward['valid'][user]):
                            continue
                        else:
                            useful_item_cnt += 1
        return [
            np.mean(result_map[topk[0]]),
            np.mean(result_ndcg[topk[0]]),
            np.mean(result_recall[topk[0]]),
            np.mean(result_map[topk[1]]),
            np.mean(result_ndcg[topk[1]]),
            np.mean(result_recall[topk[1]])
        ]


def sem_main(
    name,
    factor,
    batch_size,
    tradeoff,
    user_module,
    model_module,
    div_module,
    epoch,
    maxlen
):
    """
    主函数

    Args:
        name (`str`): 数据集名称
        factor (`int`): 隐向量维度
        batch_size (`int`): 批量大小
        tradeoff (`float`): 权衡参数
        user_module (`str`): 用户模块
        model_module (`str`): 模型模块
        div_module (`str`): 多样性模块
        epoch (`int`): 训练轮数
        maxlen (`int`): 最大序列长度
    """
    args = parse_args(
        name,
        factor,
        batch_size,
        tradeoff,
        user_module,
        model_module,
        div_module,
        epoch,
        maxlen
    )
    print(args)
    data = Data(args, 0)  # 获取数据
    meta_data = MetaData(args, data)
    session_DHRec = Train_MF(args, data, meta_data)
    session_DHRec.train()


if __name__ == "__main__":
    factor = 32
    seed = 0          
    # Gems 2048                               
    batch_size = {'Amazon_App':1024,'Kindle':1024,'Clothing':2048,'Grocery':2048,'Instant_Video':256,'Games':1024, 'ml-1m':512}
    tradeoff = {'Amazon_App':1,'Clothing':2,'Grocery':128,'Kindle':128,'Instant_Video':2,'Games':32, 'ml-1m':2}
    epoch = {'Amazon_App':15,'Kindle':5,'Clothing':10,'Grocery':20,'Instant_Video':20,'Games':10, 'ml-1m':10}
    maxlen = {'Amazon_App':5,'Kindle':3,'Clothing':3,'Grocery':5,'Instant_Video':5,'Games':5, 'ml-1m':20}

    #Setting for the proposed method and the ablations.
    #method_name:tradeoff,user_module,model_module,div_module.
    #SEM:tradeoff[data],'SAtt','dynamic','cov'.
    #w/o uDC:tradeoff[data],'static','dynamic','cov'.
    #w/o bDE:tradeoff[data],'SAtt','static','cov'.
    #w/o Div:0.0,'SAtt','dynamic','cov'.
    #w/o TPDiv:tradeoff[data],'SAtt','dynamic','AEM-cov'.

    #example:
    data = 'ml-1m'
    sem_main(data,factor,batch_size[data], tradeoff[data],'SAtt','dynamic','cov',epoch[data],maxlen[data])
