# -*- coding:utf-8 -*-
import os
import re
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import jieba
import argparse
import json
from config import category

tqdm.pandas(desc='apply')

# 加载停用词
stopwords = pd.read_csv('./data/stopwords.txt', sep='\t', header=None, names=['stopword'],
                        quoting=3, encoding='utf-8', engine='python')
STOPWORD = set(stopwords.stopword.values)


# 加载同义词林
def read_synonym_dict():
    with open('./data/synonym_forest.dict', 'r', encoding='utf-8')as  f:
        synonsym_forest = json.load(f)
    return synonsym_forest


SYNONYM_FOREST = read_synonym_dict()


def genrate_k_fold_index(category, group: dict, n=5):
    """
 生成每条数据的5-fold 的索引index
    :param category:
    :param group:
    :param n: n-flod number
    :return:
    """
    index = group[category] % n
    group[category] += 1
    return index


def matcher(string):
    '''匹配非中文字符串'''
    pattern = re.compile('[^\u4e00-\u9fff ]+')
    return set(re.findall(pattern, string))


def cut_sentences(sentences):
    """ 切词 ，默认连续非中文字符串为一个词 不包括空格"""
    sentences = sentences.replace(",", "，")
    splits = sentences.split("\t")
    total_words = []
    for sen in splits:
        match_result = matcher(sen)
        if match_result:
            for w in match_result:
                if len(w) == 1:
                    continue
                sen = sen.replace(w, f'\t->{w}\t')
            words = []
            for sub_sen in sen.split('\t'):
                if sub_sen.startswith('->'):
                    words.append(sub_sen[2:])
                    continue
                words.extend(jieba.lcut(sub_sen))
            #             print(words)
            total_words.append('\t'.join(words))
            continue
        total_words.append('\t'.join(jieba.lcut(sen)))
    return '\t\t'.join(total_words)


def df_to_csv(file, df):
    df.to_csv(file, sep=str(","), header=True, index=False, encoding='utf-8')
    print('saved ...')


def read_df_from_csv(file, names=None):
    # 重新载入数据
    data_df = pd.read_csv(file, sep=',', header=0, encoding='utf-8')
    return data_df


def is_contain_letter(str0):
    import re
    return bool(re.search('[a-zA-Z]', str0))


def query_stat(querys):
    # 提取query特征
    """

    :param querys:
    :return:
    """
    query_splits = querys.split('\t\t')
    # Query的数量
    query_num = len(query_splits)
    # Query的平均长度
    query_ave_length = 0
    # Query的最大长度
    query_max_length = 0
    # Query的最小长度
    query_min_length = 1000
    # 空格率
    blank_rate = 0
    # 字母率
    english_rate = 0

    for single_query in query_splits:
        single_query = single_query.replace('\t', '')
        query_length = len(single_query)
        query_max_length = query_length if query_length > query_max_length else query_max_length
        query_min_length = query_length if query_length < query_min_length else query_min_length
        query_ave_length += query_length
        if " " in single_query:
            blank_rate += 1
        if is_contain_letter(single_query):
            english_rate += 1
    #         print(query_length)

    query_ave_length /= query_num
    blank_rate /= query_num
    english_rate /= query_num
    stat_list = [query_num, query_max_length, query_ave_length, query_min_length, blank_rate, english_rate]

    return stat_list


def replace_synonym_word(querys):
    sens = []
    for single_query in querys.split('\t\t'):
        single_splits = single_query.split('\t')
        if len(single_splits) < 2:
            continue
        cur_words = []
        for word in single_splits:
            if word in STOPWORD:
                continue
            cur_words.append(word)
        if len(cur_words) < 2:
            continue
        cur_words = [SYNONYM_FOREST[w] if w in SYNONYM_FOREST else w for w in cur_words]

        sens.extend(cur_words + ['。'])
    return "\t".join(sens)


def remove_stopword(querys):
    ''' 去停用词'''
    words = []
    for single_query in querys.split('\t\t'):
        for word in single_query.split('\t'):
            if word in STOPWORD:
                continue
            words.append(word)
    return ' '.join(words)


# 数据读取和词分割、生成k-fold index
def preprocess(file):
    # 初始化每一子类样本总数量词典
    age_subclass_dict = defaultdict(int)
    gender_subclass_dict = defaultdict(int)
    education_subclass_dict = defaultdict(int)
    df = pd.read_csv(file, sep=',', encoding='utf-8', header=0)

    # 特征生成
    df['query_stat'] = df['Query'].progress_apply(lambda x: query_stat(x))
    df['query_num'] = df['query_stat'].progress_apply(lambda x: x[0])
    df['query_max_len'] = df['query_stat'].progress_apply(lambda x: x[1])
    df['query_ave_len'] = df['query_stat'].progress_apply(lambda x: x[2])
    df['query_min_len'] = df['query_stat'].progress_apply(lambda x: x[3])
    df['blank_rate'] = df['query_stat'].progress_apply(lambda x: x[4])
    df['english_rate'] = df['query_stat'].progress_apply(lambda x: x[5])
    # Query去停用词,且空格化
    del df['query_stat']
    df['Query'] = df['Query'].progress_apply(lambda x: replace_synonym_word(x))
    return df


def main():
    import time
    start = time.time()
    parser = argparse.ArgumentParser(description='preprocess parser')
    parser.add_argument('-i', '--inputfile', required=True, help='input file')
    parser.add_argument('-o', '--outfile', default='./data/preprocess.csv', help='output file')
    args = parser.parse_args()
    preprocess_df = preprocess(args.inputfile)
    df_to_csv(args.outfile, preprocess_df)
    """
    python data_utils.py -i ./templete/preprocess.csv -o ./data/preprocess.csv
    
    python data_utils.py -i ./templete/test_preprocess.csv -o ./data/test_preprocess.csv
    """


if __name__ == '__main__':
    main()
