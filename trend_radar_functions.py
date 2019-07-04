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
import time 

#import pickle
#with open('cleaned_twitter_df2.pkl', 'rb') as f:
#    df = pickle.load(f)  # imports cleaned dataframe
#with open('vader_output.pkl', 'rb') as f:
#    vader_output = pickle.load(f)  # imports json of vader sentiments

def reset_index(df):
    df = df.reset_index()
    return df  # returns df with index reset

def json_to_df(json_file):
    df_sentiment = json_normalize(json_file)
    return df_sentiment  # returns json in pandas df format

def combine_2_dfs(df1, df2):
    df = pd.concat([df1, df2], axis=1)
    return df  # returns a single df with the 2 dfs combined

# Combine the original df (reset index) with vader sentiment df
#df = combine_2_dfs(reset_index(df),json_to_df(vader_output))

def add_time_to_df(df):
    df['.time.'] = df['timestamp_ms'].apply(
        lambda x: time.strftime('%H:%M:%S', 
        time.gmtime(int(x)/1000)))  # add .time. column with H,M,S

# Add .time. column to the df above with sentiments
#add_time_to_df(df)

def unique_seconds_list(df):
    unique_seconds = []
    for times in df['.time.']:
        all_times = []
        all_times.append(times)
        for i in all_times:
            if not i in unique_seconds:
                unique_seconds.append(i)
    return unique_seconds  # returns list of each unique second in df

# Puts unique seconds from df into a list
#unique_sec_list = unique_seconds_list(df)

def second_groupings(seconds, seconds_list):
    second_groups = []
    for second in seconds_list:
        if len(second_groups)==0:
            second_groups.append(1)
        elif len(second_groups)%seconds != 0:
            second_groups.append(second_groups[-1])
        else:
            second_groups.append(second_groups[-1]+1)
    return second_groups # creates a list to be mapped back to the df for grouping

def seconds_dict(seconds, seconds_list):
    second_dict = dict(zip(seconds_list, second_groupings(seconds, seconds_list)))
    return second_dict  # turns list of second groupings above into dict so that it can be mapped back

def unique_words_list(df1):
    total_words = []
    for i in df1['text2']:
        words = i.split()
        for j in words:
            total_words.append(j)

    unique_words = [] 
    for i in total_words:
        if not i in unique_words:
            unique_words.append(i)
    return unique_words

def vectorize_to_df(df1):
    vectorizer = CountVectorizer(vocabulary=unique_words_list(df1))
    vectorized_words = vectorizer.transform(df1['text2'])
    word_array = vectorized_words.toarray()
    matrix_df = pd.DataFrame(word_array, columns=unique_words_list(df1), index=df1.index) 
    return matrix_df

def words_df(df1):
    df = df1.rename(index=str, columns={"text": ".text.", "lang": ".lang.", "time":".time."})
    df = df.drop('index', 1)
    df = df.reset_index()
    df = df.rename(index=str, columns={'index': 'df_index'})
    matrix_df = vectorize_to_df(df1).reset_index()
    matrix_df = matrix_df.rename(index=str, columns={'index': 'matrix_df_index'})
    df_words = df.join(matrix_df)
    return df_words

def trend_line(df1, seconds=5, sum_mean='sum'):

    if sum_mean == 'sum':
        temp_df = df1.groupby(str(seconds)+'_seconds').sum()
        temp_df = temp_df.reset_index()
    elif sum_mean == 'mean':
        temp_df = df1.groupby(str(seconds)+'_seconds').mean()
        temp_df = temp_df.reset_index()  

    plt.plot(temp_df[str(seconds)+'_seconds'], temp_df['pos'], color='red')
    plt.plot(temp_df[str(seconds)+'_seconds'], temp_df['neg'], color='grey')
    plt.xlabel('Every 5 Seconds')
    plt.ylabel('Sentiment')
    plt.title('Nintendo E3 Twitter Sentiments (sum)')
    plt.legend()
    return plt.show()