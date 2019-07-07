import json
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
import datetime
import string
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from math import pi
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def select_relevant_cols(df1, col_names=['user.id', 'text', 'lang', 'created_at', 'timestamp_ms']):
    df = df1[col_names]
    return df

def filter_lang(df1, lang='en'):
    df = select_relevant_cols(df1)
    df = df.loc[df['lang'] == lang]
    return df

def drop_duplicates(df1, subset=None, keep='first'):
    df = df1.drop_duplicates(subset=subset, keep=keep)
    return df

def unique_hashtag_list(df1):
    hashtags = []
    for text in df1['text']:
        words = text.split(' ')
        for i in words:
            if i.startswith('#'):
                hashtags.append(i)
            else:
                pass
    unique_hashtags = set(hashtags)
    unique_hashtags = list(unique_hashtags)
    return unique_hashtags

def unique_link_list(df1):
    links = []
    for text in df1['text']:
        words = text.split(' ')
        for i in words:
            if i.startswith('http'):
                links.append(i)
            else:
                pass
    unique_links = set(links)
    unique_links = list(unique_links)
    return unique_links

def unique_ats_list(df1):
    ats = []
    for text in df1['text']:
        words = text.split(' ')
        for i in words:
            if i.startswith('@'):
                ats.append(i)
            else:
                pass
    unique_ats = set(ats)
    unique_ats = list(unique_ats)
    return unique_ats

# Removes any words starting with #, http, or @ AND appends to df as 'text2'
def remove_hash_link_at(df1):
    words_to_remove_lists = [unique_hashtag_list(df1)] + [unique_link_list(df1)] + [unique_ats_list(df1)]
    words_to_remove = []
    for sublist in words_to_remove_lists:
        for i in sublist:
            words_to_remove.append(i)
    texts_final = []
    for tweet in df1['text']:
        words = tweet.split()
        resultwords = [i for i in words if i not in words_to_remove]
        result = ' '.join(resultwords)
        texts_final.append(result)
    df1['text2'] = texts_final

def strip_punctuation(tweet):
    return ''.join(c for c in tweet if c not in string.punctuation)

def remove_punctuation(df1):
    punctuation_free = []
    for tweet in df1['text2']:
        punctuation_free.append(strip_punctuation(tweet))
    df1['text2'] = punctuation_free

def make_lower_case(df1):
    lower_case = []
    for tweet in df1['text2']:
        lower_case.append(tweet.lower())
    df1['text2'] = lower_case

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_text(df1):
    lemmatizer = WordNetLemmatizer()
    lemmatized_tweets = []
    for tweet in df1['text2']:
        lemmatized = [lemmatizer.lemmatize(tweet, get_wordnet_pos(tweet)) for tweet in nltk.word_tokenize(tweet)]
        combined_words = [' '.join(lemmatized)]
        lemmatized_tweets.append(combined_words)
    lemmatized_tweets_final = [''.join(x) for x in lemmatized_tweets]
    df1['text2'] = lemmatized_tweets_final

def remove_stop_words(df1):
    stop_word_free = []
    for tweet in df1['text2']:
        words = tweet.split()
        resultwords = [i for i in words if i not in stopwords.words('english')]
        result = ' '.join(resultwords)
        stop_word_free.append(result)
    df1['text2'] = stop_word_free

def remove_just_hash(df1):
    links2 = []
    for text in df1['text2']:
        words = text.split(' ')
        for i in words:
            if i.startswith('http'):
                links2.append(i)
            else:
                pass
    unique_links2 = set(links2)
    texts_final_no_http = []
    for tweet in df1['text2']:
        words = tweet.split()
        resultwords = [i for i in words if i not in unique_links2]
        result = ' '.join(resultwords)
        texts_final_no_http.append(result)
    df1['text2'] = texts_final_no_http

def vader_sentiment(df1):
    analyzer = SentimentIntensityAnalyzer()
    vader_output = []
    for tweet in df1['text']:
        vader_output.append(analyzer.polarity_scores(tweet))
    return vader_output


