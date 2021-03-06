# -*- coding: utf-8 -*-
'''jieba

“结巴”中文分词：做最好的 Python 中文分词组件

“Jieba” (Chinese for “to stutter”) Chinese text segmentation: built to be the best Python Chinese word segmentation module.

完整文档见 README.md

GitHub: https://github.com/fxsjy/jieba

特点

支持三种分词模式：
精确模式，试图将句子最精确地切开，适合文本分析；
全模式，把句子中所有的可以成词的词语都扫描出来, 速度非常快，但是不能解决歧义；
搜索引擎模式，在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词。
支持繁体分词
支持自定义词典
MIT 授权协议
在线演示： http://jiebademo.ap01.aws.af.cm/

安装说明

代码对 Python 2/3 均兼容

全自动安装： easy_install jieba 或者 pip install jieba / pip3 install jieba
半自动安装：先下载 https://pypi.python.org/pypi/jieba/ ，解压后运行 python setup.py install
手动安装：将 jieba 目录放置于当前目录或者 site-packages 目录
通过 import jieba 来引用
'''
import sys
import os
import jieba
import numpy as np
import timeit

ChineseText ="本月29日下午，由总统秘书室室长任钟皙主持的关于特朗普访韩紧急会议在青瓦台召开。"

seg_list = jieba.cut(ChineseText, cut_all=False)
print "cut_all=False:%s" % "  ".join(seg_list)

seg_list = jieba.cut(ChineseText)
print "cut_all=Null:%s" % "  ".join(seg_list)

seg_list = jieba.cut(ChineseText, cut_all=True)
print "cut_all=True:%s" % "  ".join(seg_list)


'''读取原始语料库，然后进行分词，保存在新的目录'''
#save a string to file
def savefile(savepath, content):
    fp = open(savepath,"wb")
    #print content
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

def cutTextfiles(corpus_path = "SogouC\ClassFile", seg_path = "SogouC_seg\ClassFiel_seg"):
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
    print "File Cut finished!"




