import numpy as np
from pipeline import Pipeline
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

NUM = 3
n = 5


class ANAM(object):
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

            self.feedback = tf.placeholder(tf.int32, shape=[None, 7])  # None * 2+ 5
            self.labels = tf.placeholder(tf.float32, shape=[None,1])# none 
            self.all_attributes = tf.placeholder(tf.int32, shape=[self.n_item,self.n_attribute,NUM])
            self.attribute_EMB = tf.reduce_mean(tf.reduce_mean(tf.nn.embedding_lookup(self.weights['Q']*self.weights['Q_att'],self.all_attributes),axis=2),axis=1)#[n_item,k]
            self.item_EMB = self.weights['P']*self.weights['P_att']#[n_item,k]

            self.users_idx = self.feedback[:,0]#none
            self.items_idx = self.feedback[:,1]#none
            self.users_p5_idx = self.feedback[:,7-n:7] # none * 5
            
            self.attribute_p5 = tf.nn.embedding_lookup(self.attribute_EMB,self.users_p5_idx)# none * 5 *d
            self.item_p5 = tf.nn.embedding_lookup(self.item_EMB,self.users_p5_idx)# none * 5 * d
            
            self.item_attribute =  self.attribute_p5 * self.item_p5

            lstmCell = tf.keras.layers.LSTMCell(self.hidden_factor)
#            lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
            self.value, self.preference = tf.nn.dynamic_rnn(lstmCell, self.item_attribute, dtype=tf.float32)
                
            self.user_preference = tf.nn.embedding_lookup(self.weights['U'],self.users_idx) * self.value[:,-1,:]        
            self.item_charactor = tf.nn.embedding_lookup(self.weights['I'],self.items_idx)   

            # bias 
            self.user_preference_p5 = tf.expand_dims(tf.nn.embedding_lookup(self.weights['U'],self.users_idx),axis=1) * self.value[:,:-1,:]#[none,4,d]       
            self.item_charactor_5 = tf.nn.embedding_lookup(self.weights['I'],self.users_p5_idx)[:,1:,:]##[none,4,d]             
            self.out_p5 = tf.reduce_sum(self.user_preference_p5*self.item_charactor_5,axis=-1)#none*1

            self.out = tf.reduce_sum(self.user_preference * self.item_charactor, axis=1, keep_dims = True)#none*1

#            self.loss_rec = tf.nn.l2_loss(tf.reduce_sum(self.out,axis=1,keep_dims=True)-self.labels)
#            self.loss_rec = self.pairwise_loss(self.out,self.labels)\
#                            + self.pairwise_loss(self.out,self.labels)
            self.loss_rec = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out, labels=self.labels))\
#            +tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out_p5, labels=tf.ones_like(self.out_p5)))\
#            +tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.out_p5, labels=tf.zeros_like(self.out_p5)))\
          
            self.loss_reg = 0
            for wgt in tf.trainable_variables():
                self.loss_reg += self.lamda_bilinear * tf.nn.l2_loss(wgt)

            self.loss = self.loss_rec + self.loss_reg 
            # Optimizer.
#            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

            preference = self.user_preference

            out = tf.matmul(preference, self.weights['I'], transpose_b = True)#[none ,all]
            self.out_all_topk = tf.nn.top_k(out,1000)
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
        
        all_weights['P'] =  tf.Variable(np.random.normal(0.0, 0.001,[self.n_item, self.hidden_factor]),dtype = tf.float32) # features_M * K
        all_weights['P_att'] =  tf.Variable(np.random.normal(0.0, 0.001,[self.n_item, self.hidden_factor]),dtype = tf.float32) # features_M * K
        all_weights['Q'] =  tf.Variable(np.random.normal(0.0, 0.001,[self.num_a, self.hidden_factor]),dtype = tf.float32) # features_M * K
        all_weights['Q_att'] =  tf.Variable(np.random.normal(0.0, 0.001,[self.num_a, self.hidden_factor]),dtype = tf.float32) # features_M * K
        all_weights['U'] =  tf.Variable(np.random.normal(0.0, 0.001,[self.n_user, self.hidden_factor]),dtype = tf.float32) # features_M * K
        all_weights['I'] =  tf.Variable(np.random.normal(0.0, 0.001,[self.n_item, self.hidden_factor]),dtype = tf.float32) # features_M * K

        return all_weights

    def partial_fit(self, data):
        feed_dict = {
            self.feedback: data['feedback'],
            self.labels: data['labels'],
            self.all_attributes: data['item_attributes'],
        }
        loss_rec, loss_reg, opt = self.sess.run((self.loss_rec,self.loss_reg, self.optimizer), feed_dict=feed_dict)
        return loss_rec, 0, loss_reg

    def pairwise_loss(self,inputx,labels):
        inputx_f = inputx[1:]
        paddle = tf.expand_dims(tf.zeros(tf.shape(inputx[0])),axis=0)
        inputx_f = tf.concat([inputx_f,paddle],axis=0)
        hinge_pair = tf.maximum(tf.minimum(inputx-inputx_f,10),-10)
        loss = -tf.reduce_sum(tf.log(tf.sigmoid(hinge_pair*labels)))
        return loss
    
    def topk(self, user_item_feedback, all_attributes):
        feed_dict = {
            self.feedback: user_item_feedback,
            self.all_attributes: all_attributes
        }
        _, self.prediction = self.sess.run(self.out_all_topk, feed_dict)
        return self.prediction


class AnamTrain(Pipeline):
    """
    AnamTrain class for training the Anam model.
    """
    def __init__(self,args,data):
        super(AnamTrain, self).__init__(args,data)
        self.item_attributes = self.collect_attributes()
        self.model = ANAM(
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
