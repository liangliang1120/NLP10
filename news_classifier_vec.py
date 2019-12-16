# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 11:52:18 2019

In this part you are going to build a classifier to detect 
if a piece of news is published by the Xinhua news agency (新华社）.
1. Firstly, you have to come up with a way to represent the news. 
   (Vectorize the sentence, you can find different ways to do so online)
2. Secondly, 
   pick a machine learning algorithm that you think is suitable for this task
   
文本处理为句向量，保存结果  

@author: us
"""

import re
import jieba
import numpy as np
import pickle
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


#获取语料
df_cor = pd.read_csv('C:/Users/us/Desktop/sqlResult_1558435.csv')
df_cor = df_cor[['source','content']]
df_cor['result'] = 0
df_cor['result'][df_cor['source']=='新华社'] = 1
df_cor = df_cor.dropna(axis=0,how='any')

#使用SIF方法将句子向量化
def sen2vec(input_news,freq_dict,word_v):
    
    class Word:
            def __init__(self, text, vector):
                self.text = text
                self.vector = vector
    
    class Sentence:
        def __init__(self, word_list):
            self.word_list = word_list

        def len(self) -> int:
            return len(self.word_list)

    def get_frequency_dict(file_path):
        fileHandle = open(file_path, 'rb')
        freq_dict = pickle.load(fileHandle)
        fileHandle.close()
        return freq_dict

    def get_word_frequency(word_text, freq_dict):
        if word_text in freq_dict:
            freq = freq_dict[word_text]
            # print(freq)
            return freq
        else:
            return 1.0

    def get_word2vec(file_path):
        # print('正在载入词向量...')
        fileHandle = open(file_path, 'rb')
        word_v = pickle.load(fileHandle)
        fileHandle.close()
        return word_v

    # sentence_to_vec方法就是将句子转换成对应向量的核心方法
    def sentence_to_vec(model_v, allsent, freq_dict, embedding_size: int, a: float = 1e-3):

        sentence_set = []
        for sentence in allsent:
            vs = np.zeros(embedding_size)
            # add all word2vec values into one vector for the sentence
            sentence_length = sentence.len()
            # print(sentence.len())
            # 这个就是初步的句子向量的计算方法
            #################################################
            for word in sentence.word_list:
                # print(word.text)
                a_value = a / (a + get_word_frequency(word.text, freq_dict))
                # smooth inverse frequency, SIF
                vs = np.add(vs, np.multiply(a_value, word.vector))
                # vs += sif * word_vector

            vs = np.divide(vs, sentence_length)  # weighted average
            sentence_set.append(vs)
            # add to our existing re-calculated set of sentences
        #################################################
        # calculate PCA of this sentence set,计算主成分
        pca = PCA()
        # 使用PCA方法进行训练
        pca.fit(np.array(sentence_set))
        # 返回具有最大方差的的成分的第一个,也就是最大主成分,
        # components_也就是特征个数/主成分个数,最大的一个特征值
        u = pca.components_[0]  # the PCA vector
        # 构建投射矩阵
        u = np.multiply(u, np.transpose(u))  # u x uT
        # judge the vector need padding by wheather the number of sentences less than embeddings_size
        # 判断是否需要填充矩阵,按列填充
        if len(u) < embedding_size:
            for i in range(embedding_size - len(u)):
                # 列相加
                u = np.append(u, 0)  # add needed extension for multiplication below

        # resulting sentence vectors, vs = vs -u x uT x vs
        sentence_vecs = []
        for vs in sentence_set:
            sub = np.multiply(u, vs)
            sentence_vecs.append(np.subtract(vs, sub))
        return sentence_vecs

    def get_sentence(model_v, train='分化凸显领军自主。电动智能贯穿汽车变革。'):

        allsent = []
        for each in train:
            sent1 = list(jieba.cut(each, cut_all=False))
            # print(sent1)
            s1 = []
            for word in sent1:
                # print(word)
                try:
                    vec = model_v[word]
                except KeyError:
                    vec = np.zeros(100)
                s1.append(Word(word, vec))
            ss1 = Sentence(s1)
            allsent.append(ss1)
        return allsent

    # 获取已训练好的词向量结果
    # model_v = word_v

    # 1.构建为Sentence对象类型##########################################
    # print('句子格式处理中...')


    # 3.构建v_c全文为Sentence对象类型#############################################
    v_c = get_sentence(word_v, [input_news])

    def SIF(v_c,freq_dict):
        # v_t,content_v,v_c 按照SIF模型向量化
        # print('正在载入词频...')
        # freq_dict = get_frequency_dict('D:/开课吧/NLP11/freq_dict_File.file')

        v_c = sentence_to_vec(word_v, v_c, freq_dict, 100, 1e-3)
        return v_c

    v_c = SIF(v_c, freq_dict)
    return v_c


fileHandle = open('D:/开课吧/NLP11/freq_dict_File.file', 'rb')
freq_dict = pickle.load(fileHandle)
fileHandle.close()


fileHandle = open('D:/开课吧/NLP11/word2vec_File.file', 'rb')
word_v = pickle.load(fileHandle)
fileHandle.close()



df_cor['sen2vec'] = df_cor['content'].apply(lambda x: sen2vec(x,freq_dict,word_v))


fileHandle = open ( 'D:/开课吧/NLP11/sen2vec_news.file', 'wb' )  
pickle.dump ( df_cor, fileHandle ) 
fileHandle.close() 

# 压进文件后，读出来还是df格式
fileHandle = open('D:/开课吧/NLP11/sen2vec_news.file', 'rb')
corp = pickle.load(fileHandle)
fileHandle.close()


input_news = '''新华社北京６月２３日电（记者邢静）幸福是人们对生活最美好的期待，２０１７年哪些城市生活更幸福？２３日，“２０１７中国最具幸福感城市”调查推选活动在北京启动。
本次调查推选活动由新华社《瞭望东方周刊》与瞭望智库共同主办。该活动已连续举办十年。
本年度调查推选活动通过把“中国城市幸福感评价体系”的权重指标向“体现人民获得感”方向调整，突出“砥砺奋进·城市中国”的主题。组委会将根据新的指标体系对过去５年间的城市调查数据进行梳理，全面盘点党的十八大以来城市的发展成就和人民的生活改善状况。
自７月份起，组委会将委托专业调查机构开展入户抽样调查和利用主流新型媒体进行公众调查、大数据采集以及材料申报等工作。调查结果以公众主观调查与客观数据调查相结合，专业评价机构和评审委员会共同确认的方式产生。最终产生的“２０１７中国最具幸福感城市”名单将在年底前举办的“中国城市幸福论坛”活动上发布。
“２０１７中国最具幸福感城市”候选城市包括地级及以上城市和县级市两类。组委会在综合５年来中国最具幸福感城市榜单，并参考中国社科院等权威机构发布的“中国城市综合竞争力排名”等排名后，确定成都、杭州、南京、西安、长春、长沙、天津、苏州、北京、上海等９３个地级及以上城市进入调查推选范围。根据“２０１６全国综合实力百强县市”等名单，参考自愿申请的原则，太仓、高青、乳山、邳州、江阴、昆山、慈溪等１００个城市入围县级市候选名单。
主办方表示，为让百姓能够更多地参与到活动中来，今年将增加“中国城市幸福地标”“幸福讲堂”“‘说出你的幸福’征文”“‘幸福你就晒出来’摄影”等系列主题活动，全面解读城市居民的“幸福理念”，全方位展现人民的幸福感和获得感。
'''




