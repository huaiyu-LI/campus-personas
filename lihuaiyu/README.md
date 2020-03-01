## 主要依赖的及版本
* jieba
* scikit-learn
* fasttext
* numpy
### 数据处理 
* 使用结巴分词
* 正则匹配非中文连续字符串为一个词，所有字符均保留，暂未剔除，后续进行分析
* 针对三任务分别进行剔除未标注0分类数据,去停用词
* 同义词替换 
#### 利用 query 构建特征
* 搜索词条的数量
* 词条的平均长度、最大长度、最小长度
* 词条包含空格的比率
* 词条包含字母的比率

 

### 模型构建和评估
* 尝试使用了MultinomialNB 进行建模分析，分别对Age，Gender，Education 进行5折交叉检验，
模型检验准确度：
 <table>
        <tr>
            <th>k-fold</th>
            <th>1</th>
            <th>2</th>
            <th>3</th>
            <th>4</th>
            <th>5</th>
            <th>AVE</th>
        </tr>
        <tr>
            <th>Age</th>
            <th>0.61898</th>
            <th>0.61811</th>
            <th>0.61538</th>
            <th>0.62227</th>
            <th>0.61533</th>
            <th>0.61801</th>
        </tr>
        <tr>
             <th>Gender</th>
            <th>0.84270</th>
            <th>0.83936</th>
            <th>0.83246</th>
            <th>0.83749</th>
            <th>0.84104</th>
            <th>0.83861</th>
        </tr>
        <tr>
            <th>Education</th>
            <th>0.62840</th>
            <th>0.63642</th>
            <th>0.62801</th>
            <th>0.62914</th>
            <th>0.63449</th>
            <th>0.63129</th>
        </tr>
    </table>

## 代码执行步骤
数据分词
```shell
spark-submit preprocess_spark.py -i ./data/train.csv -o ./templete/preprocess.csv
```
数据处理，生成输入文件
```shell
python data_utils.py -i ./templete/preprocess.csv -o ./data/preprocess.csv
```
第一阶段fasttext模型训练
```shell
    python run.py -m fasttext_trian -group_name Age
    python run.py -m fasttext_trian -group_name Gender
    python run.py -m fasttext_trian -group_name Education

```
生成第一阶段特征文件
```shell
python run.py -m fasttext_feature -input_file ./data/preprocess.csv -feature_file ./data/feature.csv
```
第二阶段xgboost 调参
```shell
 python run.py -m xgb_param -group_name Age
 python run.py -m xgb_param -group_name Gender
 python run.py -m xgb_param -group_name Education

```
xgboost 模型交叉训练
```shell
 python run.py -m xgb_train -group_name Age
 python run.py -m xgb_train -group_name Gender
 python run.py -m xgb_train -group_name Education
```
flask服务
```shell 
python app.py
```





