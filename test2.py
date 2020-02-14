import tensorflow as tf
import numpy as np
import tflearn
from tensorflow.contrib.rnn import LSTMCell

class LSTM_CriticNetwork(object):
    """
    预测网络，使用词向量和动作网络中采样得到的动作得到最终的预测结果
    """
    def __init__(self,sess,dim,optimizer,learning_rate,tau,grained,max_length,dropout,wordvector):
        self.global_step=tf.Variable(0,trainable=False,name="LSTMStep")
        self.sess=sess
        self.max_length=max_length
        self.dim=dim
        self.learning_rate=tf.train.exponential_decay(learning_rate,self.global_step,10000,0.95,staircase=True)
        self.tau=tau
        self.grained=grained
        self.dropout=dropout
        self.init=tf.random_uniform_initializer(-0.05,0.05,dtype=tf.float32)
        self.L2regular=0.00001
        print("optimizer:",optimizer)
        if optimizer=="Adam":
            self.optimizer=tf.train.AdamOptimizer(self.learning_rate)
        elif optimizer=="Adagrad":
            self.optimizer=tf.train.AdagradOptimizer(self.learning_rate)
        elif optimizer=="Adadelta":
            self.optimizer=tf.train.AdadeltaOptimizer(self.learning_rate)
        self.keep_prob=tf.placeholder(tf.float32,name="keepprob")
        self.num_other_variables=len(tf.trainable_variables())

        self.wordvector=tf.get_variable("wordvector",dtype=tf.float32,initializer=wordvector,trainable=True)
        #LSTM神经元
        self.lower_cell_state,self.lower_cell_input,self.lower_cell_output,self.lower_cell_state1=self.create_LSTM_cell("Lower/Active")
        self.inputs,self.lenth,self.out=self.create_critic_network("Active")
        self.network_params=tf.trainable_variables()[self.num_other_variables:]

        self.target_wordvector=tf.get_variable("wordvector_target",dtype=tf.float32,initializer=wordvector,trainable=True)
        # LSTM神经元
        self.target_lower_cell_state,self.target_lower_cell_input,self.target_lower_cell_output,self.target_lower_cell_state1=self.create_LSTM_cell("Lower/Target")
        self.target_inputs,self.target_lenth,self.target_out=self.create_critic_network("Target")
        self.target_network_params=tf.trainable_variables()[len(self.network_params)+self.num_other_variables:]

        #延迟更新判别网络参数
        self.update_target_network_params=[self.target_network_params[i].assign(
                                            tf.multiply(self.network_params[i],self.tau)+tf.multiply(self.target_network_params[i],1-self.tau))
                                            for i in range(len(self.target_network_params))]

        self.assign_target_network_params=[self.target_network_params[i].assign(
                                                    self.network_params[i])
                                              for i in range(len(self.target_network_params)) ]

        self.assign_active_network_params=[self.network_params[i].assign(
                                                    self.target_network_params[i])
                                              for i in range(len(self.network_params))]

        self.ground_truth=tf.placeholder(tf.float32,[1,self.grained],name="ground_truth")

        #计算loss
        self.loss_target=tf.nn.softmax_cross_entropy_with_logits(labels=self.ground_truth,logits=self.target_out)
        self.loss=tf.nn.softmax_cross_entropy_with_logits(labels=self.ground_truth,logits=self.out)
        self.loss2=0
        with tf.variable_scope("Lower/Active",reuse=True):
            self.loss2+=tf.nn.l2_loss(tf.get_variable("lstm_cell/kernel"))
        with tf.variable_scope("Active/pred",reuse=True):
            self.loss2+=tf.nn.l2_loss(tf.get_variable("W"))
        self.loss+=self.loss2*self.L2regular
        self.loss_target=self.loss2*self.L2regular
        self.gradients=tf.gradients(self.loss_target,self.target_network_params)
        self.optimize=self.optimizer.apply_gradients(zip(self.gradients,self.network_params),global_step=self.global_step)

        self.num_trainable_vars=len(self.network_params)+len(self.target_network_params)












