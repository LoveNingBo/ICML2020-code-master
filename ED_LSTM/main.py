import numpy as np
import tensorflow as tf
import random
from tqdm import trange,tqdm
import sys, os
import json
import argparse
from parserr import Parser
from datamanager import DataManager
from actor import ActorNetwork
from LSTM_CRF_critic import LSTM_CriticNetwork
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#get parse
argv = sys.argv[1:]
parser = Parser().getParser()
args, _ = parser.parse_known_args(argv)
random.seed(args.seed)

"""
获取训练数据
"""
dataManager = DataManager(args.dataset)
train_data, dev_data, test_data = dataManager.getdata(args.grained, args.maxlenth)
word_vector = dataManager.get_wordvector(args.word_vector)

"""
快速测试验证代码正确性
"""
if args.fasttest == 1:
    train_data = train_data[:100]
    dev_data = dev_data[:20]
    test_data = test_data[:20]
print("train_data ", len(train_data))
print("dev_data", len(dev_data))
print("test_data", len(test_data))

def sampling_RL(sess, actor, inputs, vec, lenth, epsilon=0., Random=True):
    #print epsilon
    current_lower_state = np.zeros((1, 2*args.dim), dtype=np.float32)
    actions = []
    states = []
    # O_index=[]
    #sampling actions

    for pos in range(lenth):
        predicted = actor.predict_target(current_lower_state, [vec[0][pos]])
        
        states.append([current_lower_state, [vec[0][pos]]])
        if Random:
            if random.random() > epsilon:
                action = (0 if random.random() < predicted[0] else 1)
            else:
                action = (1 if random.random() < predicted[0] else 0)
        else:
            action = np.argmax(predicted)
        actions.append(action)
        if action == 1:
            out_d, current_lower_state = critic.lower_LSTM_target(current_lower_state, [[inputs[pos]]])

    
    Rinput = []
    for (i, a) in enumerate(actions):
        if a == 1:
            Rinput.append(inputs[i])
        # else:
        #     O_index.append(i)
    Rlenth = len(Rinput)
    if Rlenth == 0:
        actions[lenth-2] = 1
        Rinput.append(inputs[lenth-2])
        Rlenth = 1
    # Rinput += [0] * (args.maxlenth - Rlenth)
    return actions, states, Rinput, Rlenth

"""
训练过程
"""
def one_hot_vector():
    s = np.zeros(args.grained, dtype=np.float32)
    s[args.grained-1] += 1.0
    return s
def train(sess, actor, critic, train_data, batchsize, samplecnt=5, LSTM_trainable=True, RL_trainable=True):
    print("training : total ", len(train_data), "nodes.")
    # delete_loss_tensor = tf.placeholder(tf.float32,shape="delete_loss_tensor")
    random.shuffle(train_data)
    for b in range(int(len(train_data) / batchsize)):
        datas = train_data[b * batchsize: (b+1) * batchsize]
        totloss = 0.
        critic.assign_active_network()
        actor.assign_active_network()
        for j in range(batchsize):
            #prepare
            data = datas[j]
            inputs, solution, lenth = data['words'], data['solution'], data['lenth']
            # print("B_inputs:",inputs)
            # print("B_len(solution):",len(solution))
            # print("B_lenth:",lenth)
            #train the predict network
            if RL_trainable:
                actionlist, statelist, losslist = [], [], []
                aveloss = 0.
                for i in range(samplecnt):
                    actions, states, Rinput, Rlenth = sampling_RL(sess, actor, inputs, critic.wordvector_find([inputs]), lenth, args.epsilon, Random=True)
                    temp_solution=[solution[index] for index,item in enumerate(actions) if item==1 ]
                    # print("temp_solution:",temp_solution)
                    try:
                        delete_solution=np.asarray([solution[index] for index,item in enumerate(actions) if item==0])
                        # delete_solution=tf.convert_to_tensor(delete_solution)
                        O_temp=one_hot_vector()
                        delete_output= np.asarray([O_temp]*len(delete_solution))
                        delete_loss_tensor=tf.nn.softmax_cross_entropy_with_logits(labels=delete_solution, logits=delete_output)
                        # print("delete_loss_tensor :",delete_loss_tensor)
                        delete_loss=tf.Session().run(delete_loss_tensor)
                        # print("delete_loss 2:", delete_loss)
                        delete_loss=sum(delete_loss.tolist())
                    except:
                        delete_loss=0
                    # print("C_len(solution) :", len(solution))
                    actionlist.append(actions)
                    statelist.append(states)
                    out, loss = critic.getloss([Rinput], [Rlenth], [temp_solution])
                    loss=sum(loss.tolist())+delete_loss
                    loss += (float(Rlenth) / lenth) **2 *0.15
                    aveloss += loss
                    losslist.append(loss)
                
                aveloss /= samplecnt
                totloss += aveloss
                grad = None
                if LSTM_trainable:
                    temp_solution2 = [solution[index] for index, item in enumerate(actions) if item == 1]
                    out, loss, _ = critic.train([Rinput], [Rlenth], [temp_solution2])
                for i in range(samplecnt):
                    for pos in range(len(actionlist[i])):
                        rr = [0., 0.]
                        rr[actionlist[i][pos]] = (losslist[i] - aveloss) * args.alpha
                        g = actor.get_gradient(statelist[i][pos][0], statelist[i][pos][1], rr)
                        if grad == None:
                            grad = g
                        else:
                            grad[0] += g[0]
                            grad[1] += g[1]
                            grad[2] += g[2]
                actor.train(grad)
            else:
                # print("input:",inputs)
                # print("lenth:",lenth)
                # print(inputs.shape,lenth.shape)
                out, loss, _ = critic.train([inputs[:lenth]], [lenth], [solution])
                # print("solution:",solution)
                # print("out:",out)
                # print("loss:",loss.tolist())
                totloss += sum(loss.tolist())
        if RL_trainable:
            actor.update_target_network()
            if LSTM_trainable:
                critic.update_target_network()
        else:
            critic.assign_target_network()
        if (b + 1) % 500 == 0:
            acc_test = test(sess, actor, critic, test_data, noRL= not RL_trainable)
            acc_dev = test(sess, actor, critic, dev_data, noRL= not RL_trainable)
            print("batch ",b , "total loss ", totloss, "----test: ", acc_test, "| dev: ", acc_dev)

"""
测试过程
"""
def test(sess, actor, critic, test_data, noRL=False):
    acc = 0
    total=0
    for i in range(len(test_data)):
        #prepare
        data = test_data[i]
        inputs, solution, lenth = data['words'], data['solution'], data['lenth']
        total+=lenth
        #predict
        if noRL:
            out = critic.predict_target([inputs], [lenth])
            actions = [1] * lenth
        else:
            actions, states, Rinput, Rlenth = sampling_RL(sess, actor, inputs, critic.wordvector_find([inputs]), lenth, Random=False)
            out = critic.predict_target([Rinput], [Rlenth])
        # print("actions:",actions)
        O_index=[index for index,item in enumerate(actions) if item==0]
        I_index = [index for index, item in enumerate(actions) if item == 1]
        for item in O_index:
            print("solution[item]:",np.argmax(solution[item]))
            if args.grained-1==np.argmax(solution[item]):
                acc += 1
        # print("acc1:",acc)
        # print(" out:",out.shape)
        # print(" solution:",np.array(solution).shape)
        if out.shape[0]>=len(solution):
            for i in range(len(solution)):
                if np.argmax(out[i]) == np.argmax(solution[i]):
                    acc += 1
        else:
            for index,item in enumerate(I_index):
                if np.argmax(out[index]) == np.argmax(solution[item]):
                    acc += 1
        # print("acc2:",acc)
        # print("acc3:",acc/total)
    return float(acc) /  total

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config = config) as sess:
    #定义模型
    critic = LSTM_CriticNetwork(sess, args.dim, args.optimizer, args.lr, args.tau, args.grained, args.maxlenth, args.dropout, word_vector) 
    actor = ActorNetwork(sess, args.dim, args.optimizer, args.lr, args.tau)
    #打印张量
    for item in tf.trainable_variables():
        print (item.name, item.get_shape())
    
    saver = tf.train.Saver()
    
    #LSTM模型预训练
    if args.RLpretrain != '':
        pass
    elif args.LSTMpretrain == '':
        sess.run(tf.global_variables_initializer())
        for i in range(0, 5):
            train(sess, actor, critic, train_data, args.batchsize, args.sample_cnt, RL_trainable=False)
            critic.assign_target_network()
            acc_test = test(sess, actor, critic, test_data, True)
            acc_dev = test(sess, actor, critic, dev_data, True)
            print("LSTM_only ",i, "----test: ", acc_test, "| dev: ", acc_dev)
            saver.save(sess, "checkpoints/"+args.name+"_base", global_step=i)
        print("LSTM pretrain OK")
    else:
        print("Load LSTM from ", args.LSTMpretrain)
        saver.restore(sess, args.LSTMpretrain)
    
    print("epsilon", args.epsilon)

    if args.RLpretrain == '':
        for i in range(0, 5):
            train(sess, actor, critic, train_data, args.batchsize, args.sample_cnt, LSTM_trainable=False)
            acc_test = test(sess, actor, critic, test_data)
            acc_dev = test(sess, actor, critic, dev_data)
            print("RL pretrain ", i, "----test: ", acc_test, "| dev: ", acc_dev)
            saver.save(sess, "checkpoints/"+args.name+"_RLpre", global_step=i)
        print("RL pretrain OK")
    else:
        print("Load RL from", args.RLpretrain)
        saver.restore(sess, args.RLpretrain)

    for e in trange(args.epoch):
        train(sess, actor, critic, train_data, args.batchsize,  args.sample_cnt)
        acc_test = test(sess, actor, critic, test_data)
        acc_dev = test(sess, actor, critic, dev_data)
        print("epoch ", e, "----test: ", acc_test, "| dev: ", acc_dev)
        saver.save(sess, "checkpoints/"+args.name, global_step=e)

"""
DQN通过两种方式使得模型能够解决复杂问题：
1.experiment replay
2.fixed target
"""

