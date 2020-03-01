# -*- coding:utf-8 -*-
category = {
    'Age': {
        1: 0,
        2: 1,
        3: 2,
        4: 3,
        5: 4,
        6: 5},
    'Education': {
        1: 0,
        2: 1,
        3: 2,
        4: 3,
        5: 4,
        6: 5},
    "Gender": {
        1: 0,
        2: 1}
}
Elmo_Age_config = {
    'max_len': 1000,
    'batch_size': 16,
    'embedding_dims': 1024,
    'epochs': 8,
    'last_activation': 'softmax',
    'class_num': 6,
    'max_features': 512
}
Elmo_gender_config = {
    'max_len': 1000,
    'batch_size': 32,
    'embedding_dims': 1024,
    'epochs': 8,
    'last_activation': 'softmax',
    'class_num': 2
}
fasttext_config = {
    'input_file': "./data/preprocess.csv",

    'Age_train_file': "./templete/fasttext_train_Age.txt",
    'Gender_train_file': "./templete/fasttext_train_Gender.txt",
    'Education_train_file': "./templete/fasttext_train_Educatiion.txt",
    'Age_test_file': "./templete/fasttext_test_Age.txt",
    'Gender_test_file': "./templete/fasttext_test_Gender.txt",
    'Education_test_file': "./templete/fasttext_test_Educatiion.txt",

    'Age_model_dir': "./model/fasttext/fasttext_model_Age.model",
    'Gender_model_dir': "./model/fasttext/fasttext_model_Gender.model",
    'Education_model_dir': "./model/fasttext/fasttext_model_Education.model",

    'nfold_model_dir': "./model/fasttext_nfold/",
    'score_dir': './data/score/',

    'feature_file': "./data/feature.csv"

}
