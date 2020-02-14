# def binarySearch(arr,l,r,x):
#     if l<r:
#         mid=int(l+(r-l)/2)
#         if x==arr[mid]:
#             return mid
#         elif x<arr[mid]:
#             return binarySearch(arr,l,mid-1,x)
#         else:
#             return binarySearch(arr,mid+1,r,x)
#     else:
#         return -1
#
# arr = [2, 3, 4, 10, 40]
# x = 10
#
# # 函数调用
# result = binarySearch(arr, 0, len(arr) - 1, x)
# print(result)
#
# def search(arr,x):
#     for i in range(len(arr)):
#         if arr[i]==x:
#             return i
#     return -1
#
# arr = [ 'A', 'B', 'C', 'D', 'E' ];
# x = 'D';
# n = len(arr);
# result = search(arr, n, x)
# if(result == -1):
#     print("元素不在数组中")
# else:
#     print("元素在数组中的索引为", result);
#

# def insertionSort(arr):
#     for i in range(1,len(arr)):
#         temp=arr[i]
#         j=i-1
#         while arr[j]>temp and j>=0:
#             arr[j+1]=arr[j]
#             j-=1
#         arr[j+1]=temp
#
#
# arr = [12, 11, 13, 5, 6]
# insertionSort(arr)
# print(arr)
# print ("排序后的数组:")
# for i in range(len(arr)):
#     print ("%d" %arr[i])
#
#
#
#
# arr = [12, 11, 13, 5, 6]
# insertionSort(arr)
# print ("排序后的数组:")
# for i in range(len(arr)):
#     print ("%d" %arr[i])
#

# def partition(arr,l,r):
#     pivot=arr[r]
#     i=l-1
#     for j in range(l,r):
#         if arr[j]<pivot:
#             i+=1
#             arr[i],arr[j]=arr[j],arr[i]
#     arr[i+1],arr[r]=arr[r],arr[i+1]
#     return i+1
# def quicksort(arr,l,r):
#     if l<r:
#         pivot=partition(arr,l,r)
#         quicksort(arr,l,pivot-1)
#         quicksort(arr,pivot+1,r)
#
# arr = [10, 7, 8, 9, 1, 5]
# n = len(arr)
# quicksort(arr,0,n-1)
# print ("排序后的数组:")
# for i in range(n):
#     print ("%d" %arr[i]),
# def selectsort(arr):
#     for i in range(len(arr)):
#         min_j=i+1
#         for j in range(i,len(arr)):
#             if arr[j]<arr[min_j]
# name="./test.txt"
# fr = open(name)
# # n, dim = map(int, fr.readline().split())
# for i in range(4):
#     vec = fr.readline().split()
#     print("vec:",vec)
#     word = vec[0].lower()

# words=["i","like","to","watch","TV"]
# maxlenth=3
# lens = len(words)
# if maxlenth < lens:
#     print(lens)
# words += [0] * (maxlenth - lens)
# print(words)

import tensorflow as tf
import numpy as np
import tflearn

class ActorNetwork(object):
    """
    动作选择网络，根据当前地状态St对动作空间进行采样
    """
    def __init__(self,sess,dim,optimizer,learning_rate,tau):
        self.golbal_step=tf.Variable(0,trainable=False,name="ActorStep")
        self.sess=sess
        self.dim=dim
        self.learning_rate=tf.train.exponential_decay(learning_rate,self.golbal_step,10000,0.95,staircase=True)
        self.tau=tau
        if optimizer=="Adam":
            self.optimizer=tf.train.AdamOptimizer(self.learning_rate)
        elif optimizer=="Adagrad":
            self.optimizer=tf.train.AdagradOptimizer(self.learning_rate)
        elif optimizer=="Adadelta":
            self.optimizer=tf.train.AdadeltaOptimizer(self.learning_rate)
        self.num_other_variable=len(tf.trainable_variables())  #张量个数,此时还没有定义神经网络，因此此处记录的是不包含神经网络的可训练参数的个数
        #动作网络(更新)
        self.input_l,self.input_d,self.scaled_out=self.create_actor_network()
        self.network_params=tf.trainable_variables()[self.num_other_variable:]

        #动作网络(延迟更新)
        self.target_input_l,self.target_input_d,self.target_scaled_out=self.create_actor_network()
        self.target_network_params=tf.trainable_variables()[self.num_other_variable+len(self.network_params):]

        #延迟更新动作网络  ********没看懂
        self.update_target_network_params=[self.target_network_params[i].assign(
                     tf.multiply(self.network_params[i],self.tau)+tf.multiply(self.target_network_params[i],1-self.tau)
                                          )  for i in range(len(self.target_network_params))]
        self.assign_active_network_params=[self.network_params[i].assign(
                                            self.target_network_params[i])
                                            for i in range(len(self.network_params))]

        #判别网络提供的梯度
        self.action_gradient=tf.placeholder(tf.float32,[2])
        self.log_target_scaled_out=tf.log(self.target_scaled_out)

        self.actor_gradients=tf.gradients(self.log_target_scaled_out,self.target_network_params,self.action_gradient)
        print(self.actor_gradients)

        self.grads=[tf.placeholder(tf.float32,[600,1]),tf.placeholder(tf.float32,[1,]),tf.placeholder(tf.float32,[300,1])]
        self.optimize=self.optimizer.apply_gradients(zip(self.grads,self.network_params[:-1]),global_step=self.global_step)

        #定义动作采样网络
        def create_actor_network(self):
            input_l=tf.placeholder(tf.float32,shape=[1,self.dim*2])
            input_d=tf.placeholder(tf.float32,shape=[1,self.dim])

            t1=tflearn.fully_connected(input_l,1)
            t2=tflearn.fully_connected(input_d,1)

            scaled_out=tflearn.activation(   tf.matmul(input_l,t1.W)+
                                             tf.matmul(input_d,t2.W)+
                                             t1.b,
                                             activation='sigmoid')
            ###[c_{t-1} concat  h_{t-1} concat x ]代表当前的状态St

            s_out=tf.clip_by_value(scaled_out[0][0],1e-5,1-1e-5)  #tf.clip_by_value(V, min, max), 截取V使之在min和max之间

            scaled_out=tf.stack([1.0-s_out,s_out])  #结果分别为Retain\Delete的概率
            return input_l,input_d,scaled_out

        def train(self,grad):
            # feed_dict参数的作用是替换图中的某个tensor的值
            self.sess.run(self.optimize,feed_dict={self.grads[0]:grad[0],self.grads[1]:grad[1],self.grad[2]:grad[2]})

        def predict_target(self,input_l,input_d):
            return self.sess.run(self.target_scaled_out,feed_dict={self.target_input_l:input_l,
                                                                   self.target_input_d:input_d})

        def get_gradient(self,input_l,input_d,a_gradient):
            return self.sess.run(self.actor_gradients[:-1],feed_dict={
                                self.target_input_l:input_l,
                                self.target_input_d:input_d,
                                self.action_gradient:a_gradient       })

        def update_target_network(self):
            self.sess.run(self.update_target_network_params)

        def assign_active_network(self):
            self.sess.run(self.assign_active_network_params)

