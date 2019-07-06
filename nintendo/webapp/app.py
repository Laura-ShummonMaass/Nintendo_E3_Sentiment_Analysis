import random
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle
from flask import Flask, request, render_template, jsonify

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
from nintendo.trend_radar_functions import (
    reset_index, 
    json_to_df, 
    combine_2_dfs,
    add_time_to_df,
    unique_seconds_list, 
    second_groupings, 
    seconds_dict, 
    unique_words_list,
    vectorize_to_df,
    words_df,
    trend_line,
    drop_time_from_df,
    create_dictionary_for_specified_time,
    top_5_dict_to_df,
    radar_plot_creator,
    ) 

# with open('cleaned_twitter_df2.pkl', 'rb') as f:
#     df = pickle.load(f)

# with open('vader_output.pkl', 'rb') as f:
#     vader_output = pickle.load(f)

# df = combine_2_dfs(reset_index(df),json_to_df(vader_output))




with open('spam_model.pkl', 'rb') as f:
    model = pickle.load(f)
app = Flask(__name__, static_url_path="")

@app.route('/')
def index():
    """Return the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Return a random prediction."""
    data = request.json
    #prediction = model.predict_proba([data['user_input']]) #plug in html user_input id variables
    return jsonify({'hello world!'})

