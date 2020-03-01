# -*- coding:utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from joblib import dump, load
import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold, ParameterGrid
import fasttext
from model_utils import read_data, generate_cv_index, to_json
from config import fasttext_config, category
import numpy as np


class XGBoost(object):
    def __init__(self, params=None):
        self.model = self._create_model(params)

    def _create_model(self, params):
        if params is None:
            return XGBClassifier(booster='gbtree',
                                 learning_rate=0.1,
                                 n_estimators=600,  # 树的个数--1000棵树建立xgboost
                                 max_depth=8,  # 树的深度
                                 min_child_weight=1,  # 叶子节点最小权重
                                 max_delta_step=0,  # 最大增量步长，我们允许每个树的权重估计。
                                 gamma=0.,  # 惩罚项中叶子结点个数前的参数
                                 subsample=0.8,  # 随机选择80%样本建立决策树
                                 colsample_btree=0.8,  # 随机选择80%特征建立决策树
                                 objective='multi:softprob',  # 指定损失函数multi：softprob
                                 # objective='binary:logistic',
                                 # scale_pos_weight=0.25,  # 解决样本个数不平衡的问题
                                 random_state=1000,  # 随机数,
                                 num_class=len(category['Age']),
                                 reg_alpha=0,
                                 reg_lambda=5,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
                                 silent=True
                                 )
        else:
            return XGBClassifier(booster='gbtree',
                                 learning_rate=params['learning_rate'],
                                 n_estimators=params['n_estimators'],  # 树的个数--1000棵树建立xgboost
                                 max_depth=params['max_depth'],  # 树的深度
                                 min_child_weight=params['min_child_weight'],  # 叶子节点最小权重
                                 max_delta_step=0,  # 最大增量步长，我们允许每个树的权重估计。
                                 gamma=params['gamma'],  # 惩罚项中叶子结点个数前的参数
                                 subsample=params['subsample'],  # 随机选择80%样本建立决策树
                                 colsample_btree=params['colsample_bytree'],  # 随机选择80%特征建立决策树
                                 objective='multi:softprob',  # reg_lambda指定损失函数multi：softprob
                                 # objective='binary:logistic',
                                 # scale_pos_weight=0.25,  # 解决样本个数不平衡的问题
                                 random_state=1000,  # 随机数,
                                 num_class=len(category['Age']),
                                 reg_alpha=0,
                                 reg_lambda=params['reg_lambda'],  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
                                 silent=True)

    def fit(self, X, y, eval_set,
            eval_metric='mlogloss',
            early_stopping_rounds=50, verbose=None):
        self.model.fit(X, y, eval_set=eval_set,
                       eval_metric=eval_metric,
                       early_stopping_rounds=early_stopping_rounds,
                       verbose=verbose)

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    def save_model(self, path):
        dump(self.model, path)
        print(f'{path} 已保存!!! ')

    def load_model(self, path):
        self.model = load(path)

    def plot_importance_feature(self, keys=None):
        fig, ax = plt.subplots(figsize=(15, 15))

        if keys is None:
            plot_importance(self.model, height=0.5, ax=ax, max_num_features=20, )
        else:
            scores = self.model.get_booster().get_score(importance_type='weight')
            score_dict = {}
            print(scores)
            for i in range(len(keys)):
                f = f'f{i}'
                if f in scores:
                    score_dict[keys[i]] = scores[f]
            plot_importance(score_dict, height=0.5, ax=ax, max_num_features=20, )

        plt.show()

    def search(self, X, Y, params=None, n_splits=5):
        if params is None:
            params = {
                'learning_rate': [0.001, 0.01, 0.1],
                # 'max_depth': [4, 6, 8, 10]
            }
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=7)
        gsearch = GridSearchCV(estimator=self.model, return_train_score=True,
                               param_grid=params,
                               scoring='accuracy',
                               n_jobs=-1,
                               iid=False,
                               cv=kfold,
                               verbose=2
                               )
        gsearch.fit(X, Y)
        print(gsearch.param_grid)
        print(gsearch.best_score_)
        print(gsearch.best_params_)
        print(gsearch.best_estimator_)
        print(gsearch.best_index_)
        # print(gsearch.cv_results_)
        mean_train_score = gsearch.cv_results_['mean_train_score']
        std_train_score = gsearch.cv_results_['std_train_score']
        mean_test_score = gsearch.cv_results_['mean_test_score']
        std_test_score = gsearch.cv_results_['std_test_score']
        params = gsearch.cv_results_['params']
        for x, y, z, w, t in zip(params, mean_train_score, std_train_score, mean_test_score, std_test_score):
            print(f'params: {x}, mean_train_score:{y}, std_train_score:{z}, mean_test_score:{w}, std_test_score:{t}')

        return gsearch.param_grid, gsearch.best_score_, gsearch.best_params_

    def select_n_estimators(self, X, Y, cv_folds=5, early_stopping_rounds=50):
        xgb_param = self.model.get_xgb_params()
        xgtrain = xgb.DMatrix(X, label=Y)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=self.model.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='merror', early_stopping_rounds=early_stopping_rounds, verbose_eval=1)
        print(cvresult.shape[0])
        self.model.set_params(n_estimators=cvresult.shape[0])


class FastClassfier(object):
    def __init__(self, group_name):
        self.classifer = None
        self.group_name = group_name
        self.num_class = len(category[group_name])

    def train(self):
        file = fasttext_config["input_file"]
        x, y = read_data(file, group_name=self.group_name)
        if self.group_name == "Age":
            train_file = fasttext_config['Age_train_file']
            model_dir = fasttext_config['Age_model_dir']
        elif self.group_name == "Gender":
            train_file = fasttext_config['Gender_train_file']
            model_dir = fasttext_config['Gender_model_dir']
        elif self.group_name == "Education":
            train_file = fasttext_config['Education_train_file']
            model_dir = fasttext_config['Education_model_dir']
        else:
            raise Exception("请输入正确的group_name: Age,Gender,Education")

        self._generate_fasttext_data(x, y, train_file, self.group_name)
        self.classifer = fasttext.train_supervised(input=train_file,
                                                   label_prefix='__label__')

        self.classifer.save_model(model_dir)
        print(self.classifer.test(train_file))

    def save_model(self, path):
        self.classifer.save_model(path)

    def test(self, path):
        self.classifer.test(path)

    @staticmethod
    def _generate_fasttext_data(x, y, out_file, group_name):
        # out_file = out_file + "_" + group_name + ".txt"
        with open(out_file, 'w', encoding='utf-8')as f:
            for i in range(len(x)):
                # print("__label__" + str(category[group_name][y[i]]) + " , " + " ".join(x[i]))
                f.write("__label__" + str(category[group_name][y[i]]) + " , " + " ".join(x[i]) + "\n")

    def fasttext_n_fold_train(self):
        # file = "./data/preprocess.csv"
        file = fasttext_config['input_file']
        # vocab_file = './data/vocab.txt'
        x, y = read_data(file, group_name=self.group_name)
        n_fold = 0
        score = {}
        if self.group_name == "Age":
            train_file = fasttext_config['Age_train_file']
            test_file = fasttext_config['Age_test_file']
        elif self.group_name == "Gender":
            train_file = fasttext_config['Gender_train_file']
            test_file = fasttext_config['Gender_test_file']
        elif self.group_name == "Education":
            train_file = fasttext_config['Education_train_file']
            test_file = fasttext_config['Education_test_file']
        else:
            raise Exception("请输入正确的group_name, Age,Gender,Education")
        for train, test in generate_cv_index(x, y, n_splits=5):
            print(f"n-fold: {n_fold}")
            x_train = x[train]
            y_train = y[train]
            x_test = x[test]
            y_test = y[test]
            print("train_outfile:" + train_file)
            print("test_outfile: " + test_file)
            self._generate_fasttext_data(x_train, y_train, train_file, self.group_name)
            self._generate_fasttext_data(x_test, y_test, test_file, self.group_name)
            classifier = fasttext.train_supervised(input=train_file,
                                                   label_prefix='__label__')
            model_dir = fasttext_config['nfold_model_dir'] \
                        + '{}_fasttet_classifier_fold0{}.model'.format(self.group_name, n_fold)
            classifier.save_model(model_dir)
            result = classifier.test(test_file)
            score[n_fold] = result
            n_fold += 1
            print('score: ' + str(result))
        score_dir = fasttext_config['score_dir'] + '{}.json'.format(self.group_name)
        to_json(score_dir, score)

    def predict(self, x):
        label, preds = self.classifer.predict(x, k=self.num_class)

        return label, preds

    def load_model(self, path):
        self.classifer = fasttext.load_model(path)

    def get_pre_vec(self, x):
        labels, preds = self.classifer.predict(x, k=self.num_class)
        vecs = []
        for i in range(len(labels)):
            vec = np.zeros(len(labels[0]), dtype=np.float32)
            for j in range(len(labels[i])):
                # print(labels[i][j])
                label = int(labels[i][j][-1])
                vec[label] = preds[i][j]
            vecs.append(vec)
        return vecs
