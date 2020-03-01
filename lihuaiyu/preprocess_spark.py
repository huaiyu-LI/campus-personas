from pyspark.sql import SparkSession, Row
import argparse
import jieba
import re
import os
import shutil


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


def run(file, outfile, istrain):
    spark = SparkSession \
        .builder \
        .master('local[*]') \
        .appName('my App') \
        .getOrCreate()
    sc = spark.sparkContext
    rdd = sc.textFile(file)
    rdd = rdd.map(lambda x: x.split('###__###'))
    print(istrain)
    print(type(istrain))
    if istrain == 'train':
        rdd = rdd.map(lambda x: (x[0], x[1], x[2], x[3], cut_sentences(x[4])))
        personas = rdd.map(lambda p: Row(ID=str(p[0]),
                                         Age=int(p[1]),
                                         Gender=int(p[2]),
                                         Education=int(p[3]),
                                         Query=str(p[4])
                                         ))
    elif istrain == 'test':
        rdd = rdd.map(lambda x: (x[0], cut_sentences(x[1])))
        personas = rdd.map(lambda p: Row(ID=str(p[0]),
                                         Query=str(p[1])))
    else:
        raise ('istrain must be train or test!!!')
    personas_df = spark.createDataFrame(personas)
    personas_df.toPandas().to_csv(outfile, encoding='utf-8', sep=",", index=False)
    # personas_df.repartition(1).write.format('csv') \
    #     .option("header", 'true') \
    #     .option("delimiter", ",") \
    #     .save(outfile)


def main():
    import time
    start = time.time()
    parser = argparse.ArgumentParser(description='preprocess parser')
    parser.add_argument('-i', '--inputfile', default='./data/train.csv', help='input file')
    parser.add_argument('-o', '--outfile', default='./templete/preprocess.csv', help='output file')
    parser.add_argument('-op', '--opinion', default='train', help='train or test', type=str)
    args = parser.parse_args()

    run(args.inputfile, args.outfile, args.opinion)
    end = time.time()
    print('花费时间：{}'.format(end - start))


"""\
spark-submit preprocess_spark.py -i ./data/train.csv -o ./templete/preprocess.csv
spark-submit preprocess_spark.py -i ./data/test.csv -o ./templete/test_preprocess.csv -op test
"""

if __name__ == '__main__':
    main()
