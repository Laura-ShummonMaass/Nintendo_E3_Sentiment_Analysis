import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle
from flask import Flask, request, render_template, jsonify


with open('cleaned_twitter_df2.pkl', 'rb') as f:
    df = pickle.load(f)
#app = Flask(__name__, static_url_path="")

with open('vader_output.pkl', 'rb') as f:
    vader_output = pickle.load(f)
app = Flask(__name__, static_url_path="")

@app.route('/')
def index():
    """Return the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Return a random prediction."""
    data = request.json
    prediction = model.predict_proba([data['user_input']])
    return jsonify({'probability': prediction[0][1]})
