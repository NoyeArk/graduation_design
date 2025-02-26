import numpy as np
import tensorflow as tf
from pipeline import Pipeline

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

NUM = 3

class HARNN(object):
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
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        tf.reset_default_graph()  # 重置默认图
        self.graph = tf.Graph()
        with self.graph.as_default():  # , tf.device('/cpu:0'):
            # Set graph level random seed
            # Input data.
            self.weights = self._initialize_weights()
            self.neg_items = tf.placeholder(tf.int32, shape=[None, 5])
            self.feedback = tf.placeholder(tf.int32, shape=[None, 7])  # None * 2+ 5
            self.labels = tf.placeholder(tf.float32, shape=[None,1])# none 
            self.item_attributes = tf.placeholder(tf.int32, shape=[self.n_item,self.n_attribute,NUM])
            self.item_attribute = tf.reduce_mean(tf.nn.embedding_lookup(self.weights['attribute_embeddings'],self.item_attributes),axis=2)#[n_item,num_attri,k]

            self.unified_item  = tf.reduce_sum(tf.concat([self.item_attribute,tf.expand_dims(self.weights['item_embeddings'],axis=1)],axis=1),axis=1)#[n_item,k]

            self.users_idx = self.feedback[:,0]#none
            self.items_idx = self.feedback[:,1]#none
            self.item_sequence_idx = self.feedback[:,2:] # none * 5

            self.target_items = tf.concat([self.feedback[:,3:],tf.expand_dims(self.items_idx,-1)],axis=-1)#none * 5

            # pair interaction

            self.sequence = tf.nn.embedding_lookup(self.unified_item,self.item_sequence_idx)##none* p5 * d
            lstmCell = tf.keras.layers.LSTMCell(self.hidden_factor)
            value, self.preference = tf.nn.dynamic_rnn(lstmCell, self.sequence, dtype=tf.float32)#[none,5,d]
            self.preference = value
  
            self.item_pos = tf.nn.embedding_lookup(self.unified_item,self.target_items)#none *5* d
            self.item_neg = tf.nn.embedding_lookup(self.unified_item,self.neg_items)#none *5*d

            self.out_pos = tf.reduce_sum( self.preference * self.item_pos,axis=-1,keep_dims=True)#none * 1  
            self.out_neg = tf.reduce_sum( self.preference * self.item_neg,axis=-1,keep_dims=True)#none * 1  

            self.loss_rec = -tf.reduce_sum(tf.log(tf.sigmoid(self.out_pos))+tf.log(tf.sigmoid(1- self.out_neg)))

            self.loss_reg = 0
            for wgt in tf.trainable_variables():
                self.loss_reg += self.lamda_bilinear * tf.nn.l2_loss(wgt)      

            self.loss = self.loss_rec + self.loss_reg 
            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

#                self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
#                grads = self.optimizer.compute_gradients(self.loss)
#                for i, (g, v) in enumerate(grads):
#                    if g is not None:
#                        grads[i] = (tf.clip_by_norm(g, 10), v)  # clip gradients
#                self.train_op = self.optimizer.apply_gradients(grads)    
                
            elif self.optimizer_type == 'AdagradOptimizer':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

            user_embs = self.preference[:,-1,:]#none*k            
            out = tf.matmul(user_embs, self.unified_item,transpose_b=True)
            self.out_all_topk = tf.nn.top_k(out,200)

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
    
    def _initialize_weights(self):
        all_weights = dict()
        
        all_weights['user_embeddings'] =  tf.Variable(np.random.normal(0.0, 0.01,[self.n_user, self.hidden_factor]),dtype = tf.float32) # features_M * K
        all_weights['item_embeddings'] =  tf.Variable(np.random.normal(0.0, 0.01,[self.n_item, self.hidden_factor]),dtype = tf.float32) # features_M * K
        all_weights['item_embeddings_intent'] =  tf.Variable(np.random.normal(0.0, 0.01,[self.n_item, self.hidden_factor]),dtype = tf.float32) # features_M * K
        all_weights['item_att'] =  tf.Variable(np.random.normal(0.0, 0.01,[self.n_item, self.hidden_factor]),dtype = tf.float32) # features_M * K
        with tf.variable_scope('attributes'):
#            all_weights['attributes_att'] =  tf.Variable(np.random.normal(0.0, 0.01,[self.n_attribute, self.hidden_factor]),dtype = tf.float32) # features_M * K

            all_weights['attribute_embeddings'] =  tf.Variable(np.random.normal(0.0, 0.01,[self.num_a, self.hidden_factor]),dtype = tf.float32) # features_M * K
            all_weights['attribute_embeddings_intent'] =  tf.Variable(np.random.normal(0.0, 0.01,[self.num_a, self.hidden_factor]),dtype = tf.float32) # features_M * K

        return all_weights


    def partial_fit(self, data):  # fit a batch
        feed_dict = {
            self.neg_items: np.random.randint(0, self.n_item, [len(data['feedback']), 5]),
            self.feedback: data['feedback'],
            self.item_attributes: data['item_attributes']
        }
        loss_rec,loss_reg, opt = self.sess.run((self.loss_rec,self.loss_reg, self.optimizer), feed_dict=feed_dict)
        return loss_rec,0.0,loss_reg

    def pairwise_loss(self,inputx,labels):
        """
        计算pairwise的loss

        Args:
            inputx (`tensor`): shape=[batch_size, 1]

        Returns:
            loss (`tensor`): shape=[1]
        """
#        input none*1
#        label none*1
        inputx_f = inputx[1:]
        paddle = tf.expand_dims(tf.zeros(tf.shape(inputx[0])),axis=0)
        inputx_f = tf.concat([inputx_f,paddle],axis=0)
        loss = -tf.reduce_sum(tf.log(tf.sigmoid((inputx-inputx_f)*labels)))
        return loss
        
    def topk(self, user_item_feedback, item_attributes):
        feed_dict = {self.feedback: user_item_feedback, self.item_attributes:item_attributes}
        _, self.prediction = self.sess.run(self.out_all_topk,feed_dict)     
        return self.prediction


class HarnnTrain(Pipeline):
    def __init__(self, args,data):
        super(HarnnTrain, self).__init__(args,data)
        self.item_attributes = self.collect_attributes()
        self.model = HARNN(
            self.args,
            self.data,
            args['train']['factor'],
            args['train']['lr'],
            args['lamda'][args['model']],
            args['train']['optimizer']
        )

    def sample_negative(self, data, num=10):
        samples = np.random.randint( 0,self.n_item,size = (len(data)))
        return samples
