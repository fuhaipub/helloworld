# -*- coding: utf-8 -*-

'''
This module is for classifying the Chinese text.
中文语料资源来
准备：
1、中文语料库，下载地址：https://github.com/HappyShadowWalker/ChineseTextClassify

处理步骤：
Step1. 预处理，去噪声，抽取文本。此步可省略。
Step2. 中文分析，使用Python Jieba库“结巴”库，pyip上可以直接安装pip install jieba。算法RCF，基于概率的随机场算法
Step3. 构建词的向量空间。生成词袋，计算词的频率，IF-IDF = TF * IDF。
Step4.
Step5.

'''
import sys
import os
import jieba
import numpy as np
from  sklearn.datasets.base import Bunch
import cPickle as pickle
import datetime
from sklearn import feature_extraction
from sklearn.feature_extraction.text  import TfidfTransformer  #TF-IDF向量转换类
from sklearn.feature_extraction.text  import TfidfVectorizer   #TF-IDF向量生成类
from sklearn.naive_bayes import  MultinomialNB


start_time = datetime.datetime.now()

def duration():
    dur_time = "耗时：%s秒" % (datetime.datetime.now() - start_time ).seconds
    global start_time
    start_time=datetime.datetime.now()
    return dur_time

#save a string to file
def savefile(savepath, content):
    fp = open(savepath,"wb")
    fp.write(content)
    fp.close()

#read file at path, return string
def readfile(path):
    fp = open(path,"rb")
    content = fp.read()
    fp.close()
    return content

#read file at path, return list
def readlinesfile(path):
    fp = open(path,"rb")
    contents = fp.readlines()
    fp.close()
    return contents

def readbunchobj(path):
    fp = open(path, "rb")
    bunch = pickle.load(fp)
    fp.close()
    return bunch

def  writebunchobj(path,bunch):
    fp = open(path, "wb")
    pickle.dump(bunch,fp)
    fp.close()


def cutTextfiles(corpus_path , seg_path ):
    #corpus_path = "SogouC\ClassFile"
    #seg_path = "SogouC_seg\ClassFiel_seg"
    sub_path = os.listdir(corpus_path)

    for classdir in sub_path:
        file_list = os.listdir(corpus_path+"\\"+classdir)
        for textfile in file_list:
            fullname_file_org= corpus_path+"\\"+classdir+"\\"+textfile
            fullname_file_seg= seg_path+"\\"+classdir+"\\"+textfile
            content_origin = readfile(fullname_file_org).strip().replace("\r\n", "").strip()
            seg_content = jieba.cut(content_origin, cut_all=False)

            if not os.path.exists(seg_path+"\\"+classdir):
                os.makedirs(seg_path+"\\"+classdir)
            if os.path.exists(fullname_file_seg):
                os.remove(fullname_file_seg)
            savefile(fullname_file_seg, " ".join(seg_content).encode('UTF-8'))



# # ================= 构建训练集词向量空间Step1 预处理
#训练集路径
Corpus_Origin = "SogouC\ClassFile"
Corpus_Segment = "SogouC_seg\ClassFiel_seg"

wordbag_path = "train_wordbag\\train_set.dat"
if not os.path.exists("train_wordbag"):
    os.makedirs("train_wordbag")

#测试集路径
CorpusTest_Origin = "SogouCTest\ClassFile"
CorpusTest_Segment = "SogouCTest_seg\ClassFiel_seg"
wordbagtest_path = "test_wordbag\\test_set.dat"
if not os.path.exists("test_wordbag"):
    os.makedirs("test_wordbag")

#类型编码获取
catedic ={}
catlist = readlinesfile("SogouC\ClassCode.txt")
for line in catlist:
    catedic[line.strip().split(" ")[0]] = line.strip().split(" ")[1]
#print catedic


#初始化Stopwords
stpwrdlst = [word.strip() for word in readlinesfile("stopwords.txt") if word.strip() != ""]

print "Step1：预处理完成."+ duration()



# # ================= 构建词向量空间Step2 中文分词
cutTextfiles(Corpus_Origin, seg_path = Corpus_Segment)
print "Step2.1：训练集分词完成."+ duration()
cutTextfiles(CorpusTest_Origin, seg_path = CorpusTest_Segment)
print "Step2.2：测试集分词完成."+ duration()

# #=================Step3 构建训练集词向量空间
bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
catelist = os.listdir(Corpus_Segment)
bunch.target_name.extend(catelist)  #将类别信息保存在Buch对象中; extend方法，把一个列表整个追加到另一个列表中。
for mydir in catelist:
    class_path = Corpus_Segment +"\\"+mydir
    file_list = os.listdir(class_path)
    for filepath in file_list:
        fullname= class_path + "\\" + filepath
        bunch.label.append(mydir)
        bunch.filenames.append(fullname)
        bunch.contents.append(readfile(fullname).strip())

#bunch持久化
writebunchobj(wordbag_path, bunch)

print "Step3.1 : 构建训练集Bunch文本对象，保存到文件."+ duration()



bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
catelist = os.listdir(CorpusTest_Segment)
bunch.target_name.extend(catelist)  #将类别信息保存在Buch对象中; extend方法，把一个列表整个追加到另一个列表中。

for mydir in catelist:
    class_path = CorpusTest_Segment +"\\"+mydir
    file_list = os.listdir(class_path)
    for filepath in file_list:
        fullname= class_path + "\\" + filepath
        bunch.label.append(mydir)
        bunch.filenames.append(fullname)
        bunch.contents.append(readfile(fullname).strip())

#bunch持久化
writebunchobj(wordbagtest_path, bunch)
print "Step3.2 : 构建测试集Bunch文本对象，保存到文件."+ duration()


# 计算TF-IDF
# IFIDF = TFI * IDF
# TF是词在文件内部的频率分布， IDF是“逆向文件频率”，指的是词在所有文档中的所占的多少,IDF= log( 文件总数 / 词出现的文件数）
bunch = readbunchobj(wordbag_path)
tfidfspace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[], vocabulary=[])

#初始化向量空间对象
vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df =0.5)
transformer= TfidfTransformer() #该类会统计每个词语的TF-IDF权值

#文本转为词频矩阵，单独保存字典文件
tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
tfidfspace.vocabulary = vectorizer.vocabulary_

#TFDIF词袋向量的持久化
trainspace_path = "train_wordbag\\tfdifspace.dat"
writebunchobj(trainspace_path, tfidfspace)
print "Step3.3: 构建训练集TFDIF向量空间生成，保存成文件."+ duration()


#构建测试集TF-IDF向量空间
bunch = readbunchobj(wordbagtest_path)
testspace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[], vocabulary=[])

trainspace = readbunchobj(trainspace_path) #导入训练集词袋
vectorizer = TfidfVectorizer(stop_words=stpwrdlst,sublinear_tf=True,max_df=0.5,vocabulary=trainspace.vocabulary)

transformer = TfidfTransformer()
testspace.tdm = vectorizer.fit_transform(bunch.contents)
testspace.vocabulary = trainspace.vocabulary

testspace_path = "test_wordbag\\tfdifspace.dat"
writebunchobj(testspace_path, testspace)   #TFDIF词袋向量的持久化
print "Step3.4: 构建测试集TFDIF向量空间生成，保存成文件."+ duration()



#开始使用navie_bayes分类

trainspace = readbunchobj(trainspace_path)
testspace = readbunchobj(testspace_path)

clf = MultinomialNB(alpha = 0.0001).fit(trainspace.tdm, trainspace.label)

predicated = clf.predict(testspace.tdm)

total = len(predicated); rate = 0
for flabel, file_name, expct_cate in zip(testspace.label, testspace.filenames, predicated):
    if flabel != expct_cate :
        rate += 1
        print file_name, ": 实际类别：", catedic[flabel], "-->预测类别：", catedic[expct_cate]

print "error num:",rate
print "total num:",total
print "error rate:",float(rate) * 100 /float(total),"%"



