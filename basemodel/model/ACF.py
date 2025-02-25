import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

NUM = 3


class ACF(object):
    """
    基于 ACF 的推荐系统
    """
    def __init__(self, args, data, hidden_factor, learning_rate, lamda_bilinear, optimizer_type):
        # bind params to class
        self.args = args
        # bind params to class
        self.data = data
        self.n_user = self.data.entity_num['user']
        self.n_item = self.data.entity_num['item']
        self.learning_rate = learning_rate
        self.hidden_factor = hidden_factor
        self.lamda_bilinear = lamda_bilinear
        self.optimizer_type = optimizer_type
        self.n_attribute = len(self.data.item_side_entity)
        self.n_slot = self.n_attribute + 1
        # init all variables in a tensorflow graph
        np.random.seed(args.seed)

        # 属性数量
        self.num_attribute = np.sum([self.data.entity_num[key] for key in self.data.item_side_entity])
        self._init_graph()

    def _init_graph(self):
        """
        初始化一个包含输入数据、变量、模型、损失和优化器的 tensorflow 图
        """
        tf.compat.v1.reset_default_graph()  # 重置默认图
        self.graph = tf.Graph()

        with self.graph.as_default():  # , tf.device('/cpu:0'):
            self.embeddings = self._initialize_embeddings()

            # 获取输入数据
            self.feedback = tf.placeholder(tf.int32, shape=[None, 7])  # None * 2+ 5
            self.labels = tf.placeholder(tf.float32, shape=[None, 1])  # none
            self.item_attributes = tf.placeholder(tf.int32, shape=[self.n_item, self.n_attribute, NUM])

            # 物品属性
            self.item_attributes_emb = tf.reduce_mean(
                tf.nn.embedding_lookup(self.embeddings['attribute_embeddings'], self.item_attributes),
                axis=2
            )

            # 视觉特征和音频特征
            self.visacou = []

            try:
                visual = tf.constant(self.data.vis, dtype=tf.float32)
                self.visacou.append(visual)
            except AttributeError:  # 指定具体的异常类型
                print("没有视觉特征")

            try:
                acoustic = tf.constant(self.data.acou, dtype=tf.float32)
                self.visacou.append(acoustic)
            except AttributeError:  # 指定具体的异常类型
                print("没有音频特征")

            if self.visacou != []:
                self.visacou = tf.stack(self.visacou, axis=1)#[n_item,1/2,k]
                try:
                    self.item_content = tf.concat([self.item_attributes_emb, self.visacou],axis=1)  # [n_item, num_attri, k]
                except AttributeError:
                    self.item_content = self.visacou  # [n_item, num_attri, k]
            else:
                self.item_content = self.item_attributes_emb  # [n_item, num_attri, k]

            self.users_id = self.feedback[:, 0]  # none
            self.items_id = self.feedback[:, 1]  # none
            self.items_sequence_id = self.feedback[:, 2:] # none * 5

            # 对用户 ID 和物品 ID 进行嵌入
            self.users_embeddings = tf.nn.embedding_lookup(
                self.embeddings['user_embeddings'],
                self.users_id
            )  # none * k
            self.items_sequence_content = tf.nn.embedding_lookup(
                self.item_content,
                self.items_sequence_id
            )  # none * 5 * num_attri * k, 5 是物品序列长度

            # 对物品序列内容进行全连接
            self.transformed_item_sequence_content = tf.layers.dense(
                inputs=self.items_sequence_content,
                units=self.hidden_factor,
                use_bias=False
            )  # none * 5 * num_attri * k
            self.transformed_user_embeddings = tf.layers.dense(
                inputs=self.users_embeddings,
                units=self.hidden_factor
            )  # none * k

            # 将用户嵌入和物品序列内容相加
            self.user_item_combined = tf.expand_dims(
                tf.expand_dims(self.transformed_user_embeddings, 1),
                1
            ) + self.transformed_item_sequence_content  # none * 5 * num_attri * k

            # 计算注意力权重
            attention_weights = tf.nn.softmax(
                tf.layers.dense(tf.nn.relu(self.user_item_combined), 1),
                axis=2
            )  # none * 5 * num_attri * 1

            # 计算加权后的物品序列内容
            self.xl = tf.reduce_sum(attention_weights * self.items_sequence_content, axis=2)  # none * 5 * k

            # 对加权后的物品序列内容进行全连接
            self.transformed_content = tf.layers.dense(self.xl, self.hidden_factor, use_bias=False)  # none * 5 * k

            # 对物品序列内容进行嵌入
            self.items_embs1 = tf.nn.embedding_lookup(
                self.embeddings['item_embeddings1'],
                self.items_sequence_id
            )  # none * 5 * k
            self.items_embs2 = tf.nn.embedding_lookup(
                self.embeddings['item_embeddings2'],
                self.items_sequence_id
            )  # none * 5 * k

            # 将用户嵌入、加权后的物品序列内容、物品序列内容的嵌入相加
            self.F2 = tf.expand_dims(tf.layers.dense(self.users_embeddings, self.hidden_factor), 1) + \
                      self.transformed_content + \
                      tf.layers.dense(self.items_embs1, self.hidden_factor, use_bias = False) + \
                      tf.layers.dense(self.items_embs2, self.hidden_factor, use_bias = False)  # none * 5 * k

            # 计算注意力权重
            self.ail = tf.nn.softmax(tf.layers.dense(tf.nn.relu(self.F2), 1), axis=1)  # none * 5 * 1

            # 计算偏好
            self.preference = self.users_embeddings + tf.reduce_sum(self.ail * self.items_embs1, axis=1)#none * k

            # 对物品 ID 进行嵌入
            self.item_embedddings = tf.nn.embedding_lookup(self.embeddings['item_embeddings2'], self.items_id)#none * k

            # 计算偏好
            self.out = tf.reduce_sum(
                self.item_embedddings * self.preference,
                axis=-1,
                keep_dims=True
            )  # none * 1

            # 计算重构损失
            self.loss_rec = self.pairwise_loss(self.out, self.labels)

            self.loss_reg = 0
            for wgt in tf.trainable_variables():
                self.loss_reg += self.lamda_bilinear * tf.nn.l2_loss(wgt)

            self.loss = self.loss_rec + self.loss_reg
            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'AdagradOptimizer':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

            all_item_score = tf.matmul(self.preference, self.embeddings['item_embeddings2'], transpose_b=True)
            self.out_all_topk = tf.nn.top_k(all_item_score, 200)

            # init
            self.sess = self._init_session()
            init = tf.global_variables_initializer()
            self.sess.run(init)

    def _init_session(self):
        # adaptively growing video memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.allow_soft_placement = True
        return tf.Session(config=config)

    def _initialize_embeddings(self):
        """
        初始化权重

        Returns:
            embeddings (`dict`): 权重字典
        """
        embeddings = dict()
        embeddings.update({
            'user_embeddings': tf.Variable(np.random.normal(0.0, 0.01, [self.n_user, self.hidden_factor]), dtype=tf.float32),
            'item_embeddings1': tf.Variable(np.random.normal(0.0, 0.01, [self.n_item, self.hidden_factor]), dtype=tf.float32),
            'item_embeddings2': tf.Variable(np.random.normal(0.0, 0.01, [self.n_item, self.hidden_factor]), dtype=tf.float32)
        })
        with tf.name_scope('attributes'):
            embeddings['attribute_embeddings'] = tf.Variable(
                np.random.normal(0.0, 0.01, [self.num_attribute, self.hidden_factor]),
                dtype=tf.float32
            )

        return embeddings

    def partial_fit(self, feed_dict):  # fit a batch
        """
        拟合一个批次的训练数据

        Args:
            feed_dict (`dict`): 输入数据

        Returns:
            loss_rec (`float`): 重构损失
            loss_reg (`float`): 正则化损失
        """
        feed_dict = {
            self.labels: feed_dict['labels'],
            self.feedback: feed_dict['feedback'],
            self.item_attributes: feed_dict['item_attributes']
        }
        loss_rec, loss_reg, _ = self.sess.run(
            (self.loss_rec, self.loss_reg, self.optimizer),
            feed_dict=feed_dict
        )
        return loss_rec, 0.0, loss_reg

    def pairwise_loss(self, inputx, labels):
        """
        计算 pairwise loss

        Args:
            inputx (`tf.Tensor`): none*1
            labels (`tf.Tensor`): none*1

        Returns:
            loss (`tf.Tensor`): none*1
        """
        inputx_f = inputx[1:]
        paddle = tf.expand_dims(tf.zeros(tf.shape(inputx[0])), axis=0)
        inputx_f = tf.concat([inputx_f, paddle], axis=0)
        loss = -tf.reduce_sum(tf.log(tf.sigmoid((inputx - inputx_f) * labels)))
        return loss

    def topk(self, user_item_feedback, all_attributes):
        """
        计算 topk 预测

        Args:
            user_item_feedback (`np.ndarray`): 用户物品反馈
            all_attributes (`np.ndarray`): 所有属性

        Returns:
            prediction (`np.ndarray`): 预测结果
        """
        feed_dict = {
            self.feedback: user_item_feedback,
            self.item_attributes:all_attributes
        }
        _, self.prediction = self.sess.run(self.out_all_topk, feed_dict)
        return self.prediction
