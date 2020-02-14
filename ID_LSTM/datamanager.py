import numpy as np
import tensorflow as tf
import json, random

"""
数据预处理
"""
# class DataManager(object):
#     def __init__(self, dataset):
#         '''
#         Read the data from dir "dataset"
#         '''
#         self.origin = {}
#         for fname in ['train', 'dev', 'test']:
#             data = []
#             print("Loading {} dataset...".format(fname))
#             # n=0
#             for line in open('%s/%s.res' % (dataset, fname)):
#                 # n+=1
#                 # if fname=="test":
#                 #     print(n)
#                 s = json.loads(line.strip())
#                 if len(s) > 0:
#                     data.append(s)
#             self.origin[fname] = data
#     """
#     获取单词字典
#     """
#     def getword(self):
#         '''
#         Get the words that appear in the data.
#         Sorted by the times it appears.
#         {'ok': 1, 'how': 2, ...}
#         Never run this function twice.
#         '''
#         wordcount = {}
#         def dfs(node):
#             if node.__contains__('children'):
#                 dfs(node['children'][0])
#                 dfs(node['children'][1])
#             else:
#                 word = node['word'].lower()
#                 wordcount[word] = wordcount.get(word, 0) + 1
#         for fname in ['train', 'dev', 'test']:
#             for sent in self.origin[fname]:
#                 dfs(sent)
#         words = wordcount.items()
#         sorted(words,key = lambda x : x[1], reverse = True)
#         self.words = words
#         self.wordlist = {item[0]: index+1 for index, item in enumerate(words)}
#         return self.wordlist
#     """
#     读取训练、验证、测试数据
#     """
#     def getdata(self, grained, maxlenth):###grained代表分类的类别
#         '''
#         Get all the data, divided into (train,dev,test).
#         For every sentence, {'words':[1,3,5,...], 'solution': [0,1,0,0,0]}
#         For each data, [sentence1, sentence2, ...]
#         Never run this function twice.
#         '''
#         def one_hot_vector(r):
#             s = np.zeros(grained, dtype=np.float32)
#             s[r] += 1.0
#             return s
#         def dfs(node, words):
#             if node.__contains__('children'):
#                 dfs(node['children'][0], words)
#                 dfs(node['children'][1], words)
#             else:
#                 word = self.wordlist[node['word'].lower()]
#                 words.append(word)
#         self.getword()
#         self.data = {}
#         temp_max_length=0
#         for fname in ['train', 'dev', 'test']:
#             self.data[fname] = []
#             for sent in self.origin[fname]:
#                 words = []
#                 dfs(sent, words)
#                 lens = len(words)
#                 if maxlenth < lens:
#                     print(lens)
#                 if lens>temp_max_length:
#                     temp_max_length=lens
#                 words += [0] * (maxlenth - lens)
#                 print("words:",words)
#                 """
#               将文本类别转换为one-hot编码
#               [0,1,0,0] 文本类别为2
#               训练数据长度
#               data['train']：[{'words': ,'solution': ,'lenth': },{},{}]
#                """
#                 solution = one_hot_vector(int(sent['rating']))
#                 now = {'words': np.array(words),\
#                         'solution': solution,\
#                         'lenth': lens}
#                 self.data[fname].append(now)
#         # print("数据集中最长句长为：{}".format(temp_max_length))
#         return self.data['train'], self.data['dev'], self.data['test']
#     """
#     读取词向量
#     """
#     def get_wordvector(self, name):
#         fr = open(name,"r",encoding="utf-8")
#         n=2000000#len(fr.readlines())
#         # print("Glove词向量行数:{}".format(n))
#         dim =300# map(int, fr.readline().split())
#         self.wv = {}
#         for i in range(n - 1):
#             try:
#                 vec = fr.readline().split()
#             except:
#                 break
#             # print("vec[:10]:",vec[:10])
#             if len(vec[1:])==300:
#                 word = vec[0].lower()
#                 # print("word:",word)
#                 vec = list(map(float, vec[1:]))
#             else:
#                 continue
#             if self.wordlist.__contains__(word):
#                 self.wv[self.wordlist[word]] = vec
#         print("Glove词向量有效大小：{}".format(len(self.wv)))
#         self.wordvector = []
#         losscnt = 0
#         for i in range(len(self.wordlist) + 1):
#             if self.wv.__contains__(i):
#                 self.wordvector.append(self.wv[i])
#             else:
#                 losscnt += 1
#                 self.wordvector.append(np.random.uniform(-0.1,0.1,[dim]))
#         self.wordvector = np.array(self.wordvector, dtype=np.float32)
#         print(losscnt, "words not find in wordvector")
#         print(len(self.wordvector), "words in total")
#         return self.wordvector

# datamanager = DataManager("../TrainData/MR")
# train_data, test_data, dev_data = datamanager.getdata(2, 200)
# wv = datamanager.get_wordvector("../WordVector/vector.25dim")
# mxlen = 0
# for item in train_data:
#    print item['lenth']
#    if item['lenth'] > mxlen:
#        mxlen =item['lenth']
# print mxlen

""""""

class DataManager(object):
    def __init__(self, dataset):
        '''
        Read the data from dir "dataset"
        '''
        self.origin = {}
        for fname in ['train', 'valid', 'test']:
            data = []
            print("Loading {} dataset...".format(fname))
            # n=0
            for line in open('%s/%s' % (dataset, fname)):
                # if fname=="test":
                #     print("Loading {fname} Dataset...")
                s = line.strip()
                if len(s) > 0:
                    data.append(s)
            self.origin[fname] = data
        self.intent={}
        for line in open('%s/vocab.intent' % (dataset)):
            s = line.strip()
            self.intent[s]=len(self.intent)
    """
    获取单词字典
    """

    """
    读取训练、验证、测试数据
    """
    def getdata(self, grained, maxlenth):###grained代表分类的类别
        '''
        Get all the data, divided into (train,dev,test).
        For every sentence, {'words':[1,3,5,...], 'solution': [0,1,0,0,0]}
        For each data, [sentence1, sentence2, ...]
        Never run this function twice.
        '''
        def one_hot_vector(r):
            s = np.zeros(grained, dtype=np.float32)
            s[r] += 1.0
            return s

        print("Getting word list...")
        wordcount = {}
        for fname in ['train', 'valid', 'test']:
            for sent in self.origin[fname]:
                words = []
                slot_tag_line, _ = sent.strip('\n\r').split(' <=> ')
                if slot_tag_line == "":
                    continue
                for item in slot_tag_line.split(' '):
                    tmp = item.split(":")
                    assert len(tmp) >= 2
                    word, tag = tmp[:-1], tmp[-1]
                    if len(word)==1:
                        word=word[0].lower()
                    else:
                        print("\nError :多个单词！")
                    wordcount[word] = wordcount.get(word, 0) + 1
        words = wordcount.items()
        sorted(words,key = lambda x : x[1], reverse = True)
        self.words = words
        self.wordlist = {item[0]: index+1 for index, item in enumerate(words)}

        print("Processing train dataset...")
        self.data = {}
        temp_max_length=0
        for fname in ['train', 'valid', 'test']:
            print("Processing {} dataset ...".format(fname))
            self.data[fname] = []
            for sent in self.origin[fname]:
                # print("sent:",sent)
                words = []
                slot_tag_line, class_name = sent.strip('\n\r').split(' <=> ')
                class_name=class_name.split(';')[0]
                if slot_tag_line == "":
                    continue
                # in_seq, tag_seq = [], []
                for item in slot_tag_line.split(' '):
                    tmp = item.split(":")
                    assert len(tmp) >= 2
                    word, tag = tmp[:-1], tmp[-1]
                    lowercase = True
                    # print("word:{}".format(word))
                    if len(word)==1:
                        word=word[0]
                    else:
                        print("\nError :多个单词！")
                    if lowercase:
                        word = self.wordlist[word.lower()]
                    words.append(word)
                    # wordcount[word] = wordcount.get(word, 0) + 1
                lens = len(words)

                if maxlenth < lens:
                    print(lens)
                if lens>temp_max_length:
                    temp_max_length=lens
                words += [0] * (maxlenth - lens)
                solution = one_hot_vector(int(self.intent[class_name]))
                # print(words)
                now = {'words': np.array(words), \
                       'solution': solution, \
                       'lenth': lens}
                self.data[fname].append(now)
        print("数据集中最长句长为：",temp_max_length)
        # words = wordcount.items()
        # sorted(words,key=lambda x: x[1], reverse=True)
        # words.sort(key=lambda x: x[1], reverse=True)
        # self.words = words
        # self.wordlist = {item[0]: index + 1 for index, item in enumerate(words)}
        return self.data['train'], self.data['valid'], self.data['test']

        # self.data = {}
        # for fname in ['train', 'dev', 'test']:
        #     self.data[fname] = []
        #     for sent in self.origin[fname]:
        #         words = []
        #         dfs(sent, words)
        #         lens = len(words)
        #         if maxlenth < lens:
        #             print(lens)
        #         words += [0] * (maxlenth - lens)
        #         """
        #       将文本类别转换为one-hot编码
        #       [0,1,0,0] 文本类别为2
        #       训练数据长度
        #       data['train']：[{'words': ,'solution': ,'lenth': },{},{}]
        #        """
        #         solution = one_hot_vector(int(sent['rating']))
        #         now = {'words': np.array(words),\
        #                 'solution': solution,\
        #                 'lenth': lens}
        #         self.data[fname].append(now)
        # return self.data['train'], self.data['dev'], self.data['test']
    """
    读取词向量
    """
    def get_wordvector(self, name):
        fr = open(name,"r",encoding="utf-8")
        n=2000000#len(fr.readlines())
        # print("\nn:{}".format(n))
        # exit("停止")
        dim = 300#map(int, fr.readline().split())
        self.wv = {}
        for i in range(n - 1):
            try:
                vec = fr.readline().split()
            except:
                break
            # print(vec[:10])
            # if len(vec[1:])==300:
            #     word = vec[0].lower()
            #     vec = map(float, vec[1:])
            # else:
            #     continue
            if len(vec[1:]) == 300:
                # continue
                word = vec[0].lower()
                vec = list(map(float, vec[1:]))
            else:
                continue
            if self.wordlist.__contains__(word):
                self.wv[self.wordlist[word]] = vec
        self.wordvector = []
        losscnt = 0
        for i in range(len(self.wordlist) + 1):
            if self.wv.__contains__(i):
                self.wordvector.append(self.wv[i])
            else:
                losscnt += 1
                self.wordvector.append(np.random.uniform(-0.1,0.1,[dim]))
        self.wordvector = np.array(self.wordvector, dtype=np.float32)
        print(losscnt, "words not find in wordvector")
        print(len(self.wordvector), "words in total")
        return self.wordvector