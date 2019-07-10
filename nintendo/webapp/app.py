import random
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle
from flask import Flask, request, render_template, jsonify, make_response

import json
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
import datetime
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import matplotlib.pyplot as plt
from math import pi
import time 
from nintendo.functions import (
    trend_2018_select_secs
    ) 

# with open('cleaned_twitter_df2.pkl', 'rb') as f:
#     df = pickle.load(f)

# with open('vader_output.pkl', 'rb') as f:
#     vader_output = pickle.load(f)

# df = combine_2_dfs(reset_index(df),json_to_df(vader_output))



# with open('spam_model.pkl', 'rb') as f:
#     model = pickle.load(f)
app = Flask(__name__, static_url_path="")

# trend_line = trend_line_for_web(start_time_str='16:07:24', 
#                                 end_time_str='16:10:36',
#                                 sum_mean='sum')

@app.route('/')
def index():
    """Return the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Return a random prediction."""
    data = request.json
    
    print("after_2")

    #prediction = model.predict_proba([data['start_time'], data['end_time']]) #plug in html user_input id variables
    filename = trend_2018_select_secs(start_time=data['start_time'], 
                                    end_time=data['end_time'],
                                    sum_mean='sum')
    print("after_1")
    # return prediction
    return f'<img src="/tmp/{filename}" width="400" height="320"/>' #class="chart" />'

# @app.route('/trend_line_time.png', methods=['GET', 'POST'])
# def trend_line():
    # '''Return trend line'''
    # #data = request.json
    # with open ('/Users/laurashummonmaass/Documents/flatiron/capstone_final/Nintendo_E3_Sentiment_Analysis/nintendo/webapp/static/tmp/trend_line_time.png', 'rb') as f:
    #     image_data = f.read()
    # response = make_response(image_data)
    # response.headers.set('Content-Type', 'image/png')
    # return response 

