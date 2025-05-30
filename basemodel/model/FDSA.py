import numpy as np
import tensorflow.compat.v1 as tf
from modules import ff, positional_encoding, multihead_attention

tf.disable_v2_behavior()

from pipeline import Pipeline

NUM = 3


class FDSA(object):
    """
    基于 FDSA 的推荐系统
    """
    def __init__(self,args,data,hidden_factor, learning_rate, lamda_bilinear, optimizer_type):
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
        np.random.seed(args['seed'])
        self.num_a = np.sum([self.data.entity_num[key] for key in self.data.item_side_entity])
        self._init_graph()

    def _init_graph(self):
        '''
        初始化一个包含输入数据、变量、模型、损失和优化器的 tensorflow 图
        '''
        tf.reset_default_graph()  # 重置默认图
        self.graph = tf.Graph()
        with self.graph.as_default():  # , tf.device('/cpu:0'):
            self.weights = self._initialize_weights()

            self.feedback = tf.placeholder(tf.int32, shape=[None, 7])  # None * 2+ 5
            self.labels = tf.placeholder(tf.float32, shape=[None,1])  # none
            self.neg_items = tf.placeholder(tf.int32, shape=[None, 5])
            self.item_attributes = tf.placeholder(tf.int32, shape=[self.n_item, self.n_attribute, NUM])

            self.users_idx = self.feedback[:,0]  # none
            self.items_idx = self.feedback[:,1]  # none
            self.target_items = tf.concat(
                [
                    self.feedback[:, 3:],
                    tf.expand_dims(self.items_idx, -1)
                ],
                axis=-1
            )  # none * 5
            self.item_sequence = self.feedback[:, 2:] # none * 5

            # 物品属性
            feature_embs = tf.reduce_mean(
                tf.nn.embedding_lookup(
                    self.weights['attribute_embeddings'],
                    self.item_attributes
                ),
                axis=2
            )  # all * num_attri * k

            # 物品序列
            self.item_sequence_embs = tf.nn.embedding_lookup(
                self.weights['sequence_embeddings'],
                self.item_sequence
            )  # none * 5 * k

            # 物品属性序列
            self.feature_sequence_embs = tf.nn.embedding_lookup(feature_embs,self.item_sequence)#none *5 * feature * d
            att_feature = tf.nn.softmax(tf.layers.dense(self.feature_sequence_embs,self.hidden_factor),axis=2)
            self.feature_sequence_embs = tf.reduce_sum(self.feature_sequence_embs * att_feature,axis=2)#none *5* d

            F = self.feature_sequence_embs + self.weights['position']#none *5* d
            S =self.item_sequence_embs #+ self.weights['position']#none *5* d
            self.O_s =self.SAB( S,'item')#none *5* d
            self.O_f =self.SAB( F,'feature')#none *5* d

            O_sf = tf.concat([self.O_s,self.O_f],axis=-1)#none *5* 2d
            
            O_sf = tf.layers.dense(O_sf,self.hidden_factor) #none *5* d        
            
            O_sft =tf.expand_dims( O_sf[:,-1,:],axis=1)#none*1 * d   
            
            self.item_pos =  tf.nn.embedding_lookup(self.weights['item_embeddings'],self.target_items)#none *5* d
            self.item_neg =  tf.nn.embedding_lookup(self.weights['item_embeddings'],self.neg_items)#none *5*d
            
            self.out_pos =  tf.reduce_sum( O_sft * self.item_pos,axis=-1,keep_dims=True)#none * 1  
            self.out_neg =  tf.reduce_sum( O_sft * self.item_neg,axis=-1,keep_dims=True)#none * 1  
            
            self.loss_rec = -tf.reduce_sum(tf.log(tf.sigmoid(self.out_pos))+tf.log(tf.sigmoid(1- self.out_neg)))

#            self.loss_rec = self.pairwise_loss(self.out,self.labels)\
#                            + self.pairwise_loss(self.out,self.labels)
            self.loss_reg = 0
            for wgt in tf.trainable_variables():
                self.loss_reg += self.lamda_bilinear * tf.nn.l2_loss(wgt)      

            self.loss = self.loss_rec+ self.loss_reg 
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

            out = tf.matmul(O_sf[:,-1,:],self.weights['item_embeddings'],transpose_b=True)
            self.out_all_topk = tf.nn.top_k(out,200)
            # init
            self.sess = self._init_session()
            init = tf.global_variables_initializer()
            self.sess.run(init)

    def jieduan(self, x, name):
        """
        计算密集层

        Args:
            x (`np.ndarray`): 输入数据
            name (`str`): 名称

        Returns:
            `np.ndarray`: 密集层
        """
        return tf.layers.dense(x,self.hidden_factor,use_bias=False)

    def scaled_dot_product_attention(self, queries, keys, values, name):
        """
        计算 scaled dot product attention

        Args:
            queries (`np.ndarray`): 查询
            keys (`np.ndarray`): 键
            values (`np.ndarray`): 值
            name (`str`): 名称

        Returns:
            `np.ndarray`: scaled dot product attention
        """
        self.num_heads = 4
        batch_size, num_queries, sequence_length = tf.shape(queries)[0], tf.shape(queries)[1], tf.shape(values)[1]
        Q, K, V = self.jieduan(queries,name), self.jieduan(keys,name), self.jieduan(values,name)
        Q = tf.transpose(tf.reshape(Q, [batch_size, num_queries, self.num_heads, int(self.hidden_factor/self.num_heads)]), [0, 2, 1, 3])
        K = tf.transpose(tf.reshape(K, [batch_size, sequence_length, self.num_heads, int(self.hidden_factor/self.num_heads)]), [0, 2, 1, 3])
        V = tf.transpose(tf.reshape(V, [batch_size, sequence_length, self.num_heads, int(self.hidden_factor/self.num_heads)]), [0, 2, 1, 3])
        S = tf.matmul(tf.nn.softmax(tf.matmul(Q, tf.transpose(K, [0, 1, 3, 2])) / tf.sqrt(float(self.hidden_factor))), V)
        S = tf.reshape(tf.transpose(S, [0, 2, 1, 3]), [batch_size, num_queries, int(self.hidden_factor)])
        return S#tf.keras.layers.LayerNormalization(axis=-1)(S)#tf.stop_gradient(S)* tf.Variable(np.ones([1,1,self.hidden_factor]),dtype=tf.float32)

    def SAB(self,sequence_embs,name):
        """
        计算 scaled dot product attention

        Args:
            sequence_embs (`np.ndarray`): 输入数据
            name (`str`): 名称

        Returns:
            `np.ndarray`: scaled dot product attention
        """
        # sequence_embs # none *5* d
        M_f = self.scaled_dot_product_attention(sequence_embs,sequence_embs,sequence_embs,name)
        LayerNormalization = tf.keras.layers.LayerNormalization()
        M_f = LayerNormalization(M_f+sequence_embs)
        O_f = tf.layers.dense(tf.layers.dense(M_f,self.hidden_factor),self.hidden_factor,activation=tf.nn.relu)
        O_f = LayerNormalization(O_f+sequence_embs)
        return O_f

    def encode(self, xs, training=True):
        """
        编码

        Args:
            xs (`np.ndarray`): 输入数据
            training (`bool`): 是否训练

        Returns:
            `np.ndarray`: encoder outputs. (N, T1, d_model)
        """
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            x, seqlens, sents1 = xs

            # src_masks
            src_masks = tf.math.equal(x, 0) # (N, T1)

            # embedding
            enc = tf.nn.embedding_lookup(self.embeddings, x) # (N, T1, d_model)
            enc *= self.hp.d_model**0.5 # scale

            enc += positional_encoding(enc, self.hp.maxlen1)
            enc = tf.layers.dropout(enc, self.hp.dropout_rate, training=training)

            ## Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              key_masks=src_masks,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False)
                    # feed forward
                    enc = ff(enc, num_units=[self.hp.d_ff, self.hp.d_model])
        memory = enc
        return memory, sents1, src_masks
    
    def _init_session(self):
        # adaptively growing video memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.allow_soft_placement = True
        return tf.Session(config=config)
    
    def _initialize_weights(self):
        """
        初始化权重

        Returns:
            `dict`: 权重字典
        """
        all_weights = dict()
        all_weights['attribute_embeddings'] =  tf.Variable(np.random.normal(0.0, 0.001,[self.n_attribute, self.hidden_factor]),dtype = tf.float32) # features_M * K
        all_weights['item_embeddings'] =  tf.Variable(np.random.normal(0.0, 0.001,[self.n_item, self.hidden_factor]),dtype = tf.float32) # features_M * K
        all_weights['sequence_embeddings'] =  tf.Variable(np.random.normal(0.0, 0.001,[self.n_item, self.hidden_factor]),dtype = tf.float32) # features_M * K
        all_weights['position']  =  tf.Variable(np.random.normal(0.0, 0.001,[1,5, self.hidden_factor]),dtype = tf.float32) #1 *5* d

        return all_weights

    def partial_fit(self, data):  # fit a batch
        neg = np.random.randint(0,self.n_item,np.shape(data['feedback'][:,2:]))
        feed_dict = {
            self.feedback: data['feedback'],
            self.item_attributes: data['item_attributes'],
            self.neg_items: neg
        }
        loss_rec,loss_reg, opt = self.sess.run((self.loss_rec,self.loss_reg, self.optimizer), feed_dict=feed_dict)
        return loss_rec,0,loss_reg

    def topk(self, user_item_feedback, item_attributes):
        feed_dict = {
            self.feedback: user_item_feedback,
            self.item_attributes: item_attributes
        }
        _, self.prediction = self.sess.run(self.out_all_topk, feed_dict)
        return self.prediction

    def pairwise_loss(self,inputx,labels):
        """
        计算 pairwise loss

        Args:
            inputx (`np.ndarray`): 输入数据
            labels (`np.ndarray`): 标签

        Returns:
            `float`: pairwise loss
        """
        A = - tf.log(tf.sigmoid(inputx) * (labels*2-1) + tf.abs(1-labels))
        return tf.reduce_sum(A)  # tf.nn.l2_loss(inputx-labels)


class FdsaTrain(Pipeline):
    def __init__(self,args,data):
        super(FdsaTrain, self).__init__(args, data)
        self.item_attributes = self.collect_attributes()
        self.model = FDSA(
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
