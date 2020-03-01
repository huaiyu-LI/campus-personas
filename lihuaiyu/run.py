# -*- coding:utf-8 -*-
from model import XGBoost, FastClassfier
from data_utils import *
from model_utils import *
from config import *
import warnings
from sklearn.metrics import accuracy_score
import argparse

warnings.filterwarnings("ignore")

import time


# --------------------------------------------------------------------------------------
def fasttext_train(group_name=None):
    classifier = FastClassfier(group_name)
    classifier.fasttext_n_fold_train()
    classifier.train()


def _get_vec(x, group_name):
    classifier = FastClassfier(group_name)
    model_dir = fasttext_config[f'{group_name}_model_dir']
    classifier.load_model(model_dir)
    # test = "柔和 双沟 。 中财网 首页 财经 。 周公 解梦 万事俱备 查询 2345 。 曹云金 再讽 郭德纲"
    vecs = classifier.get_pre_vec(x)
    return vecs


def fasttext_gen_vec_file(in_file, out_file, mode='train'):
    df = pd.read_csv(in_file, sep=',', encoding='utf-8', header=0)
    print(len(df))
    if mode == 'train':
        df = df.query('Age > 0 and Gender > 0 and Education > 0')
        print(len(df))
        df = df.dropna(how='any', axis=0)
    df["Query"] = df['Query'].progress_apply(lambda x: str(x).replace("\t \t", "\t").replace("\t", " "))
    for group_name in category.keys():
        vecs = _get_vec(df.Query.values.tolist(), group_name=group_name)
        for i in range(len(category[group_name])):
            vec_list = [vec[i] for vec in vecs]
            df[f'{group_name}_vec_{i}'] = vec_list
    print(df.head(10))
    print(df.keys())
    if mode == 'train':
        # names = ['ID', 'Query', 'age_kfold_index', 'gender_kfold_index',
        #          'education_kfold_index', 'query_stat']
        names = ['ID', 'Query']
    else:
        names = ['Query']
    df = df.drop(labels=names, axis=1)
    df_to_csv(out_file, df)


# --------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------
def xgb_train(group_name="Age"):
    train_data, labels = read_xgb_data(group_name=group_name)
    params = read_json(f'./model/params/xgb_{group_name}_params.dict')
    print(params)
    model = XGBoost(params)
    print(model.model.get_xgb_params())
    print(labels)
    model.select_n_estimators(train_data, labels)
    accs = []
    cnt = 1
    for train_index, test_index in generate_cv_index(train_data, labels):
        x_train = train_data[train_index, :]
        y_train = labels[train_index]
        x_test = train_data[test_index]
        y_test = labels[test_index]
        print(y_test)
        print(type(y_test))
        model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=True)
        y_pred = model.predict(x_test)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
        accs.append(accuracy)
        cnt += 1
    accs.append(np.array(accs).mean())
    to_json(f'./data/score/{group_name}_xgb_cv', accs)
    # model.fit(x, y, eval_set=[(x,y)], verbose=True)
    model.save_model(f'./model/xgb/xgb_{group_name}_.model')


def xgb_predict(test_file):
    test_df, test_data = read_xbg_test_data(test_file)
    model = XGBoost()
    group_names = ["Age", 'Gender', 'Education']
    for group_name in group_names:
        model.load_model(f'./model/xgb/xgb_{group_name}_.model')
        y_preds = model.predict(test_data)
        predictions = [round(value) + 1 for value in y_preds]
        # res_df = pd.DataFrame({'preds': predictions})
        test_df[group_name] = predictions
    df_to_csv('./data/test_result.csv', test_df)


def gen_best_xbg_params(group_name='Age'):
    import time
    start = time.time()
    train_data, labels = read_xgb_data(group_name=group_name)
    train_data, labels = generate_samples_data(train_data, labels, group_name=group_name, sample_num=3000)

    model = XGBoost()
    # 选择初始最优n_estimators
    model.select_n_estimators(train_data, labels)
    # 确定max_depth和min_weight参数
    param_grid = {'max_depth': [2, 4, 6, 8],
                  'min_child_weight': [1, 2, 3, 4, 5]}
    _, _, best_param = model.search(train_data, labels, params=param_grid)
    model.model.set_params(max_depth=best_param['max_depth'])
    model.model.set_params(min_child_weight=best_param['min_child_weight'])
    # 确定gamma参数
    param_grid = {'gamma': [1, 2, 3, 4, 5, 6, 7, 8, 9]}
    _, _, best_param = model.search(train_data, labels, params=param_grid)
    model.model.set_params(gamma=best_param['gamma'])
    # 调整subsample与colsample_bytree参数
    param_grid = {'subsample': [i / 10.0 for i in range(5, 11)],
                  'colsample_bytree': [i / 10.0 for i in range(5, 11)]}
    _, _, best_param = model.search(train_data, labels, params=param_grid)
    model.model.set_params(subsample=best_param['subsample'])
    model.model.set_params(colsample_bytree=best_param['colsample_bytree'])
    # 调整正则
    param_grid = {'reg_lambda': [i / 10.0 for i in range(1, 11)]}
    _, _, best_param = model.search(train_data, labels, params=param_grid)
    model.model.set_params(reg_lambda=best_param['reg_lambda'])
    # 学习率
    param_grid = {'learning_rate': [0.001, 0.01, 0.1]}
    _, _, best_param = model.search(train_data, labels, params=param_grid)
    model.model.set_params(learning_rate=best_param['learning_rate'])
    print(model.model.get_xgb_params())
    to_json(f'./model/params/xgb_{group_name}_params.dict', model.model.get_xgb_params())
    end = time.time()
    print('cost time: {}'.format(end - start))


def single_predict(query):
    query = cut_sentences(query)
    feauture = query_stat(query)
    query = replace_synonym_word(query)
    return feauture, query


# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', required=True, help='')
    parser.add_argument('-input_file', help='')
    parser.add_argument('-feature_file', help='')
    parser.add_argument('-group_name', help='')
    args = parser.parse_args()
    if args.mode == 'fasttext_trian':
        fasttext_train(group_name=args.group_name)
    elif args.mode == 'fasttext_feature':
        fasttext_gen_vec_file(args.input_file, args.feature_file)
    elif args.mode == 'xgb_param':
        gen_best_xbg_params(group_name=args.group_name)
    elif args.mode == 'xgb_train':
        xgb_train(group_name=args.group_name)
    elif args.mode == 'xgb_test':
        xgb_predict(test_file=args.feature_file)

    """
    python run.py -m fasttext_trian -group_name Age
    python run.py -m fasttext_trian -group_name Gender
    python run.py -m fasttext_trian -group_name Education
    
    python run.py -m fasttext_feature -input_file ./data/preprocess.csv -feature_file ./data/feature.csv
    
    python run.py -m xgb_param -group_name Age
    python run.py -m xgb_param -group_name Gender
    python run.py -m xgb_param -group_name Education
    
    python run.py -m xgb_train -group_name Age
    python run.py -m xgb_train -group_name Gender
    python run.py -m xgb_train -group_name Education
    """


if __name__ == '__main__':
    main()
#
