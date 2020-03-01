from flask import Flask, render_template, url_for, request
from data_utils import *
from model import *

app = Flask(__name__)
classifier_age = FastClassfier('Age')
classifier_gender = FastClassfier('Gender')
classifier_edu = FastClassfier('Education')
classifier_age.load_model(fasttext_config['Age_model_dir'])
classifier_gender.load_model(fasttext_config['Gender_model_dir'])
classifier_edu.load_model(fasttext_config['Education_model_dir'])
model_age = XGBoost()
model_gender = XGBoost()
model_edu = XGBoost()
model_age.load_model('model/xgb/xgb_Age_.model')
model_gender.load_model('model/xgb/xgb_Gender_.model')
model_edu.load_model('model/xgb/xgb_Education_.model')


@app.route('/')
def home():
    return render_template('home.html')


def single_predict(query):
    query = cut_sentences(query)
    feauture = query_stat(query)
    query = replace_synonym_word(query)
    return feauture, query


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        feature, query = single_predict(message)
        preds = classifier_age.get_pre_vec([query])
        feature.extend(preds[0].tolist())
        preds = classifier_edu.get_pre_vec([query])
        feature.extend(preds[0].tolist())
        preds = classifier_gender.get_pre_vec([query])
        feature.extend(preds[0].tolist())
        result = {}
        result['Age'] = model_age.predict(feature)[0] + 1
        result['Gender'] = model_gender.predict(feature)[0] + 1
        result['Education'] = model_edu.predict(feature)[0] + 1
    else:
        raise ('wrong request method, please use POST')
    return render_template('result.html', prediction=result)


if __name__ == '__main__':
    app.run( debug=False)
