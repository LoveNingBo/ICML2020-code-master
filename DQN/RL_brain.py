import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)

class DeepQNetwork:
    #创建网络
    def _build_net(self):
        self.s=tf.placeholder(tf.float32,[None,self.n_features],name="s")#用来接收observation 即state
        self.q_target=tf.placeholder(tf.float32,[None,self.n_actions],name="Q_target")
        # 用来接收q_target的值，DQN神经网络输入是状态observation（state)，输出是每个action对应的Q值，
        # 那么有几个action,神经网络就会输出几个Q值
        #tf.variable_scope()指定作用域进行区分  tf.GraphKeys包含所有graph collection中的标准集合名
        with tf.variable_scope("eval_net"):
            c_names,n_l1,w_initializer,b_initializer=["eval_net_params",tf.GraphKeys.GLOBAL_VARIABLES],10,tf.random_normal_initializer(0,0.3),\
            tf.constant_initializer(0.1)
            #函数原型：tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=None, dtype=tf.float32)

            #eval_net第一层
            with tf.variable_scope("l1"):
                w1=tf.get_variable("w1",[self.n_features,n_l1],initializer=w_initializer,collections=c_names)
                b1=tf.get_variable("b1",[1,n_l1],initializer=b_initializer,collections=c_names)
                l1=tf.nn.relu(tf.matmul(self.s,w1)+b1)
            #eval_net第二层
            with tf.variable_scope("l2"):
                w2=tf.get_variable("w2",[n_l1,self.n_actions],initializer=w_initializer,collections=c_names)
                b2=tf.get_variable("b2",[1,self.n_actions],initializer=b_initializer,collections=c_names)
                self.q_eval=tf.matmul(l1,w2)+b2
        #求误差
        with tf.variable_scope("loss"):
            self.loss=tf.reduce_mean(tf.squared_difference(self.q_target,self.q_eval))
        #梯度下降
        with tf.variable_scope("train"):
            self._train_op=tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)


        """创建target 神经网络 ，提供Q_target"""
        self.s_=tf.placeholder(tf.float32,[None,self.n_actions],name="s_") #接受下一状态observation (state)
        with tf.get_variable_scope("target_net"):                         #tensorflow的collection提供一个全局的存储机制，不会受到变量名生存空间的影响。
            c_names=["target_net_params",tf.GraphKeys.GLOBAL_VARIABLES]   #一处保存，到处可取。

            #target_net的第一层
            with tf.variable_scope("l1"):
                w1=tf.get_variable("w1",[self.n_features,n_l1],initializer=w_initializer,collections=c_names)
                b1=tf.get_variable("b1",[1,n_l1],initializer=b_initializer,collections=c_names)
                l1=tf.nn.relu(tf.matmul(self.s_,w1)+b1)

            #target_net第二层
            with tf.variable_scope("l2"):
                w2=tf.get_variable("w2",[self.l1,self.n_actions],initializer=w_initializer,collections=c_names)
                b2=tf.get_variable("b2",[1,self.n_actions],initializer=b_initializer,collections=c_names)
                self.q_next=tf.matmul(l1,w2)+b2

    #参数初始化
    def __init__(self,n_actions,n_features,learning_rate=0.01,reward_decay=0.9,e_greedy=0.9,replace_target_iter=300,memory_size=500,batch_size=32,e_greedy_increment=None,output_graph=False):
        self.n_actions=n_actions
        self.n_features=n_features
        self.lr=learning_rate
        self.gamma=reward_decay
        self.epsilon_max=e_greedy
        self.replace_target_iter=replace_target_iter
        self.memory_size=memory_size
        self.batch_size=batch_size
        self.epsilon_increment=e_greedy_increment
        self.epsilon=0 if e_greedy_increment is not None else self.epsilon_max #是否开启探索模式，并且逐渐减少

        self.learn_step_counter=0

        #初始化全0记忆 [s,a,r,s_]
        self.memory=np.zeros((self.memory_size,n_features*2+2))

        #创建[target_net,eval_net]
        self._build_net()

        #替换target_net中的参数
        t_params=tf.get_collection("target_net_params") #提取target_net中的参数
        e_params=tf.get_collections("eval_net_params")

        self.replace_target_op=[tf.assign(t,e) for t,e in zip(t_params,e_params)]

        self.sess=tf.Session()

        if output_graph:
            tf.summary.FileWriter("logs/",self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his=[]

    #存储记忆
    def store_transition(self,s,a,r,s_):
        #hasattr()函数用于判断对象是否包含对应的属性。
        if not hasattr(self,"memory_counter"):
            self.memory_counter=0
        transition=np.hstack((s,[a,r],s_))

        index=self.memory_counter%self.memory_size
        self.memory[index,:]=transition
        self.memory_counter+=1

    #选择行为action
    def choose_action(self,observation):
        observation=observation[np.newaxis,:]

        if np.random.uniform()<self.epsilon:
            actions_value=self.sess.run(self.q_eval,feed_dict={self.s:observation})
            action=np.argmax(actions_value)
        else:
            action=np.random.randint(0,self.n_actions)
        return action

    def learn(self):
        #检查是否替换target_net的参数
        if self.learn_step_counter%self.replace_target_iter==0:
            self.sess.run(self.replace_target_op)
            print("\ntarget_params_replaced.")

        #从记忆库中采样，去除数据之间的相关性,如果存储的记忆超过记忆库大小，那么就在记忆库大小的数组内采样
        #如果当前存储的记忆没有填满记忆库，那么就在已存储的数据中进行采样
        if self.memory_counter>self.memory_size:
            sample_index=np.random.choice(self.memory_size,size=self.batch_size)
        else:
            sample_index=np.random.choice(self.memory_counter,size=self.batch_size)
        batch_memory=self.memory[sample_index,:]

        #获取下一时刻的q_eval和q_target，其中q_eval是由实时更新的q_eval_net计算得到
        #而q_target是由延迟更新参数的q_target_net计算得到，利用结构相同但是参数更新不同步的机制去掉相关性
        """
       q_target是q_target_net获得，q_eval是由q_eval_net获得 
       """
        q_next,q_eval=self.sess.run([self.next,self.q_eval],feed_dict={self.s_:batch_memory[:,-self.n_features:],
                                                                             self.s:batch_memory[:,:self.n_features]})
        q_target=q_eval.copy()
        batch_index=np.arange(self.batch_size,dtype=np.int32)       #[n_features,a,r,n_features]对应[s,a,r,s_]
        eval_act_index=batch_memory[:,self.n_features].astype(int)  #一个batch中动作action对应的索引
        reward=batch_memory[:,self.n_features+1]
        q_target[batch_index,eval_act_index]=reward+self.gamma*np.max(q_next,axis=1)

        #训练eval_net
        _,self.cost=self.sess.run([self._train_op,self.loss],feed_dict={self.s:batch_memory[:,:self.n_features],self.q_target:q_target})
        self.cost_his.append(self.loss)

        #增加epsilon
        self.epsilon=self.epsilon+self.epsilon_increment if self.epsilon<self.epsilon_max else self.epsilon_max
        self.learn_step_counter+=1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)),self.cost_his)
        plt.ylabel("Cost")
        plt.xlabel("training steps")
        plt.show()



































































