from collections import Counter
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import json

tqdm.pandas(desc='apply')


def data_file(file, outfile):
    word_freq = defaultdict(int)
    lines = []
    writer = open(outfile, 'w', encoding='utf-8')

    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            splits = line.strip().split(' ')
            if splits[0].endswith("@"):
                continue
            lines.append(splits)
            for w in splits:
                word_freq[w] += 1
    word_freq_sorted = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    polysemant = set()
    for key, value in word_freq_sorted:
        if value >= 2:
            polysemant.add(key)
    for word_list in lines:
        tmp_w_list = []
        for i in range(len(word_list)):
            if i == 0:
                tmp_w_list.append(word_list[i])
                continue
            if word_list[i] in polysemant:
                continue
            tmp_w_list.append(word_list[i])
        if len(tmp_w_list) <= 2:
            continue
        writer.write(" ".join(tmp_w_list) + "\n")


def word_stat(querys, word_freq):
    for query in querys.strip().split('\t\t'):
        word_freq.update(query.split('\t'))


def load_synonym_forest(file):
    """读取同义词林"""
    synonym_list = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            tmp_list = line.strip().split(" ")
            del tmp_list[0]
            synonym_list.append(tmp_list)
    return synonym_list


def builder(path, out_path):
    synonyms = load_synonym_forest('./data/cilin_filtered.txt')
    print(synonyms)
    # w = open(out_path, 'w', encoding='utf-8')
    word_freq = Counter()
    # new_synonyms = []
    synonyms_dict = {}
    df = pd.read_csv(path, sep=',', header=0, encoding='utf-8')
    print(df.head(10))
    df.Query.progress_apply(lambda x: word_stat(x, word_freq))
    print(word_freq)
    for single_synonym in synonyms:
        replaced_word = ""
        replaced_word_num = 0
        for word in single_synonym:
            tmp_num = word_freq[word]
            if tmp_num > replaced_word_num:
                replaced_word_num = tmp_num
                replaced_word = word
        if replaced_word == "":
            replaced_word = single_synonym[0]
        for word in single_synonym:
            if word == replaced_word:
                continue
            print(replaced_word, word)
            synonyms_dict[replaced_word] = word
        single_synonym_str = " ".join(single_synonym)
        # w.write(f'{replaced_word},{single_synonym_str}\n')
        # new_synonyms.append((replaced_word, single_synonym))
    # w.close()
    to_json(out_path, synonyms_dict)
    pass


def to_json(file, data_dict):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, ensure_ascii=False)


def read_json():
    with open('./data/synonym_forest.dict', 'r', encoding='utf-8')as  f:
        synonsym_forest = json.load(f)
    return synonsym_forest


def main():
    # in_file = "./data/cilin.txt"
    # out_file = "./data/cilin_filtered.txt"
    # data_file(in_file, out_file)
    path = './data/preprocess.csv'
    # out_path = './data/synonym.dict'
    out_path = './data/synonym_forest.dict'
    builder(path, out_path)
    result = read_json()
    for key, value in result.items():
        print(key, value)


if __name__ == '__main__':
    main()
