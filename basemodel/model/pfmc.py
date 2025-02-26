import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from pipeline import Pipeline


class PFMC(object):
    """
    PFMC模型
    """
    def __init__(self,args,data,hidden_factor, learning_rate, lamda_bilinear, optimizer_type):
        self.args = args
        self.data = data
        self.n_user = self.data.entity_num['user']
        self.n_item = self.data.entity_num['item']
        self.learning_rate = learning_rate
        self.hidden_factor = hidden_factor
        self.lamda_bilinear = lamda_bilinear
        self.optimizer_type = optimizer_type
        self.loss_type = 'square_loss'

        np.random.seed(args['seed'])

        self._init_graph()

    def _init_graph(self):
        """
        初始化一个包含输入数据、变量、模型、损失函数和优化器的 tensorflow 图。
        """
        tf.reset_default_graph()  # 重置默认图
        self.graph = tf.Graph()

        with self.graph.as_default():  # , tf.device('/cpu:0'):
            # 输入数据，None * 2 + 5
            self.feedback = tf.placeholder(tf.int32, shape=[None, 7])
            self.labels = tf.placeholder(tf.float32, shape=[None, 1])

            self.weights = self._initialize_weights()

            # 用户和物品的索引
            self.users_idx = self.feedback[:, 0]  # none
            self.items_idx = self.feedback[:, 1]  # none
            self.users_p5_idx = self.feedback[:, 7 - self.args['neighbor_cnt']: 7]  # none * 5
            self.l1_loss = []

            # 用户和物品的嵌入
            self.UI = tf.nn.embedding_lookup(self.weights['UI'], self.users_idx)  # none*k
            self.IU = tf.nn.embedding_lookup(self.weights['IU'], self.items_idx)  # none*k
            self.out1 = tf.reduce_sum(self.UI * self.IU, axis=1, keep_dims=True)

            print('self.users_p5_idx:', self.users_p5_idx)
            print('self.weights["IL"]:', self.weights['IL'].shape)

            # 用户和物品的嵌入
            self.IL = tf.reduce_mean(tf.nn.embedding_lookup(self.weights['IL'], self.users_p5_idx), axis=1)  # none*k
            self.LI = tf.nn.embedding_lookup(self.weights['LI'], self.items_idx)#none*k
            self.out2 = tf.reduce_sum(self.IL * self.LI, axis=1, keep_dims=True)

            # 用户和物品的嵌入
            # self.UL = tf.nn.embedding_lookup(self.weights['UI'],self.users_idx)#none*k
            # self.LU = tf.reduce_mean(tf.nn.embedding_lookup(self.weights['LU'],self.users_p5_idx),axis=1)#none*k

            self.out = self.out1 + self.out2  # none * 1            
            self.loss_rec = self.pairwise_loss(self.out,self.labels)
            # self.loss_l1 = tf.reduce_sum(tf.stack(self.l1_loss))tf.Variable(0,dtype=tf.float32)

            self.loss_reg = 0
            for wgt in tf.trainable_variables():
                self.loss_reg += self.args['lamda'][self.args['model']] * tf.nn.l2_loss(wgt)
            self.loss = self.loss_rec + self.loss_reg

            # Optimizer
            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate
                ).minimize(self.loss)
            elif self.optimizer_type == 'AdagradOptimizer':
                self.optimizer = tf.train.AdagradOptimizer(
                    learning_rate=self.learning_rate
                ).minimize(self.loss)
            elif self.optimizer_type == 'GradientDescentOptimizer':
                self.optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate=self.learning_rate
                ).minimize(self.loss)
            elif self.optimizer_type == 'MomentumOptimizer':
                self.optimizer = tf.train.MomentumOptimizer(
                    learning_rate=self.learning_rate,
                    momentum=0.95
                ).minimize(self.loss)

            self.score1= tf.matmul(self.UI,tf.transpose(self.weights['IU']))
            self.score2 = tf.matmul(self.IL,tf.transpose(self.weights['LI']))
            self.score = self.score1 + self.score2
            self.out_all_topk = tf.nn.top_k(self.score,200)
            self.trainable = tf.trainable_variables()

            # init
            self.sess = self._init_session()
            init = tf.global_variables_initializer()
            self.sess.run(init)

    #For model
    def _init_session(self):
        # adaptively growing video memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.7
        return tf.Session(config=config)

    def _initialize_weights(self):
        """
        初始化权重
        """
        all_weights = dict()
        for key in self.data.entity:
            n_entity = self.data.entity_num[key]
            all_weights[key + '_embeddings'] = tf.Variable(
                np.random.normal(
                    0.0,
                    0.01,
                    [n_entity, self.hidden_factor]
                ),
                dtype = tf.float32
            ) # features_M * K

        all_weights['UI'] = tf.Variable(np.random.normal(0.0, 0.01, [self.n_user, self.hidden_factor]), dtype = tf.float32)  # features_M * K
        all_weights['IU'] = tf.Variable(np.random.normal(0.0, 0.01, [self.n_item, self.hidden_factor]), dtype = tf.float32)  # features_M * K
        all_weights['IL'] = tf.Variable(np.random.normal(0.0, 0.01, [self.n_item, self.hidden_factor]), dtype = tf.float32)  # features_M * K
        all_weights['LI'] = tf.Variable(np.random.normal(0.0, 0.01, [self.n_item, self.hidden_factor]), dtype = tf.float32)  # features_M * K
        all_weights['UL'] = tf.Variable(np.random.normal(0.0, 0.01, [self.n_user, self.hidden_factor]), dtype = tf.float32)  # features_M * K
        all_weights['LU'] = tf.Variable(np.random.normal(0.0, 0.01, [self.n_item, self.hidden_factor]), dtype = tf.float32)  # features_M * K

        return all_weights

    def partial_fit(self, data):
        """
        训练一个batch。

        Args:
            data (`dict`): 包含feedback和labels的字典。

        Returns:
            loss_rec (`float`): 正样本的损失。
            loss_reg (`float`): 正则化损失。
        """
        feed_dict = {
            self.feedback: data['feedback'],
            self.labels: data['labels']
        }
        loss_rec, loss_reg, opt = self.sess.run(
            (self.loss_rec, self.loss_reg, self.optimizer),
            feed_dict=feed_dict
        )
        return loss_rec, loss_reg

    def pairwise_loss(self,inputx,labels):
        """
        计算 pairwise 损失
        """
        inputx_f = inputx[1:]
        inputx_f = tf.concat([inputx_f,tf.zeros([1,1])],axis=0)
        loss = -tf.reduce_sum(tf.log(tf.sigmoid((inputx-inputx_f)*labels)))
        return loss

    def topk(self, user_item_feedback):
        """
        获取 topk 预测

        Args:
            user_item_feedback (`np.ndarray`): 用户和物品的反馈

        Returns:
            prediction (`np.ndarray`): 预测结果
        """
        feed_dict = {self.feedback: user_item_feedback}
        _, self.prediction = self.sess.run(self.out_all_topk,feed_dict)
        return self.prediction


class PfmcTrain(Pipeline):
    def __init__(self, args, data):
        super(PfmcTrain, self).__init__(args, data)
        self.model = PFMC(
            self.args,
            self.data,
            args['train']['factor'],
            args['train']['lr'],
            args['lamda'][args['model']],
            args['train']['optimizer']
        )

    def sample_negative(self, data, num=10):
        samples = np.random.randint(0, self.n_item, size=len(data))
        return samples
