# -*- coding:utf-8 -*-
import json
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import numpy as np
from config import *


def read_json(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data_dict = json.load(f)
    return data_dict


def to_json(json_path, data_dict):
    with open(json_path, 'w') as f:
        json.dump(data_dict, f)
    print(f'{json_path} 已保存！')


def read_data(file, group_name="Age"):
    df = pd.read_csv(file, sep=',', encoding='utf-8', header=0)
    df = df.dropna(axis=0, how='any')
    df = df[df[group_name] != 0]
    contents = [query.replace("\t \t", "\t").split("\t") for query in df['Query'].values]
    return np.array(contents), np.array(df[group_name].values)


def generate_cv_index(x, y, n_splits=5):
    sfolder = StratifiedKFold(n_splits=n_splits, random_state=1024, shuffle=True)
    return sfolder.split(x, y)


def read_xgb_data(group_name):
    df = pd.read_csv(fasttext_config['feature_file'], sep=',', encoding='utf-8', header=0)
    labels = df[group_name].values.tolist()
    labels = [category[group_name][i] for i in labels]
    names = ["Age", "Gender", "Education"]
    # print(df.groupby('Age').agg('count'))
    data = df.drop(labels=names, axis=1).values
    labels = np.array(labels)
    return data, labels


def read_xbg_test_data(file):
    df = pd.read_csv(file, sep=',', encoding='utf-8', header=0)

    return df[['ID']], df.drop(labels='ID', axis=1).values


def generate_samples_data(data, labels, group_name, sample_num=100):
    df = pd.DataFrame({'label': labels.tolist(), 'data': data.tolist()})
    samples = []
    samples_labels = []
    for index in range(len(category[group_name])):
        sub_df = df[df.label == index]
        sub_data = sub_df.data.values.tolist()
        sub_label = sub_df.label.values.tolist()
        if len(sub_data) > sample_num:
            sub_data = sub_data[:sample_num]
            sub_label = sub_label[:sample_num]
        samples.extend(sub_data)
        samples_labels.extend(sub_label)
    return np.array(samples), np.array(samples_labels)


def vocab_builder(file, vocab_file, vocab_size=100000):
    df = pd.read_csv(file, sep=',', encoding='utf-8', header=0)
    df = df.dropna(axis=0, how='any')
    all_data = []
    for content in df.Query.values[:100]:
        all_data.extend(content.split('\t'))
    counter = Counter(all_data)

    count_pairs = counter.most_common(vocab_size - 1)
    print()
    words = list(list(zip(*count_pairs))[0])
    print(words)
    words = ['<PAD>'] + words
    with open(vocab_file, 'w', encoding='utf-8') as w:
        w.write("\n".join(words) + "\n")


def read_vocab(vocab_file):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open(vocab_file, 'r', encoding='utf-8') as fp:
        # 如果是py2 则每个值都转化为unicode
        words = [_.strip() for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def encode_cate(content, words, vocab_size=100000):
    """将id表示的内容转换为文字"""
    return [words[x] if x in words else vocab_size for x in content]


def encode_sentences(contents, words):
    """将id表示的内容转换为文字"""
    return [encode_cate(x, words) for x in contents]


def pad_sent(x, max_len=Elmo_Age_config['max_len']):
    if len(x) > max_len:
        return x[:max_len]
    else:
        return x + [''] * (max_len - len(x))


def batch_generator(x, y, batch_size=Elmo_Age_config['batch_size']):
    n_batcher_per_epoch = len(x) // batch_size
    for i in range(n_batcher_per_epoch):
        x_batch = e.sents2elmo([pad_sent(sent) for sent in x[batch_size * i:batch_size * (i + 1)]])
        y_batch = y[batch_size * i: batch_size * (i + 1)]
        yield np.array(x_batch), y_batch


if __name__ == '__main__':
    file = "./data/preprocess.csv"
    vocab_file = './data/vocab.txt'
    x, y = read_data(file)
    # print(len(x))
    # print(len(y))
    # print(type(x[0]))
    # print(type(generate_cv_index(x, y)))
    # vocab_builder(file, vocab_file)
    import time

    #
    for batch_x, batch_y in batch_generator(x, y):
        time.sleep(2)
        print(batch_x, batch_y)
