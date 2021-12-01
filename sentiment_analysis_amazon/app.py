# from flask import Flask, jsonify,  request, render_template
# from sklearn.externals import joblib
# import numpy as np
from flask import Flask, render_template, request
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.decomposition import TruncatedSVD
import model

app = Flask(__name__)
# model_load = joblib.load("./models/nlp_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/recommend", methods=['POST'])
def recommend():
    if (request.method == 'POST'):
        username = request.form['User_Name']
        rc = model.rcmd(username)
        # if type(rc) == type('string'):
        #     m_str = rc
        # else:
        #     m_str = "---".join(rc)
            # return m_str
        return render_template('index.html', column_names=rc.columns.values, row_data=list(rc.values.tolist()), zip=zip)
    else:
        return render_template('index.html')

# @app.route("/recommend", methods=["POST"])
# def similarity():
#     if (request.method == 'POST'):
#         username = request.form['User_Name']
#         rc = rcmd(username)
#         if type(rc) == type('string'):
#             m_str = rc
#         else:
#             m_str = "---".join(rc)
#             # return m_str
#         return render_template('index.html', recommend_text='Recommended Products {}'.format(m_str))
#     else:
#         return render_template('index.html')
#
# @app.route("/recommend_api", methods=['POST', 'GET'])
# def predict_api():
#     print(" request.method :",request.method)
#     if (request.method == 'POST'):
#         data = request.get_json()
#         return jsonify(model_load.predict([np.array(list(data.values()))]).tolist())
#     else:
#         return render_template('index.html')


if __name__ == '__main__':
    app.debug=True
    app.run()
