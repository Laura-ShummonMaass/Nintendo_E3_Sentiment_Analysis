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
import pickle
import os

from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def reset_index(df):
    '''Resets df index.'''
    df = df.reset_index()
    return df  

def json_to_df(json_file):
    '''Turns a json file into a df.'''
    '''Also un-embeds dictionaries.'''
    df_sentiment = json_normalize(json_file)
    return df_sentiment  

def combine_2_dfs(df1, df2):
    '''Concatenates 2 dataframes.'''
    df = pd.concat([df1, df2], axis=1)
    return df  

def add_time_to_df(df1):
    '''Adds a .time. column in string format'''
    df1['.time.'] = df1['timestamp_ms'].apply(
        lambda x: time.strftime('%H:%M:%S', 
                                time.gmtime(int(x)/1000)))
    return df1  

def drop_time_from_df(df1):
    '''Drops all .time. columns from df.'''
    df1 = df1.drop('.time.', axis=1)
    return df1  

def unique_seconds_list(df):
    '''Creates a list of every unique second in the dataframe's .time. col.'''
    unique_seconds = []
    for times in df['.time.']:
        all_times = []
        all_times.append(times)
        for i in all_times:
            if not i in unique_seconds:
                unique_seconds.append(i)
    return unique_seconds  

def second_groupings(seconds, seconds_list):
    '''Returns a list with len equal to the rows of the dataframe.'''
    '''Assigns each tweet to a group dependant on it's time.'''
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
    '''Returns a dictionary using the groupings above that can easily be added'''
    '''into the dataframe.'''
    second_dict = dict(zip(seconds_list, second_groupings(seconds, seconds_list)))
    return second_dict  # turns list of second groupings above into dict so that it can be mapped back

def unique_words_list(df1):
    '''Creates a list of every unique word in the text2 column of the df.'''
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
    '''Creates a dataframe of sentiments (each word is a col)'''
    vectorizer = CountVectorizer(vocabulary=unique_words_list(df1))
    vectorized_words = vectorizer.transform(df1['text2'])
    word_array = vectorized_words.toarray()
    matrix_df = pd.DataFrame(word_array, columns=unique_words_list(df1), index=df1.index) 
    return matrix_df

def words_df(df1):
    '''Merges the words dataframe to the original df.'''
    df = df1.rename(index=str, columns={"text": ".text.", "lang": ".lang.", "time":".time."})
    df = df.drop('index', 1)
    df = df.reset_index()
    df = df.rename(index=str, columns={'index': 'df_index'})
    matrix_df = vectorize_to_df(df1).reset_index()
    matrix_df = matrix_df.rename(index=str, columns={'index': 'matrix_df_index'})
    df_words = df.join(matrix_df)
    return df_words

def trend_line(df1, seconds=5, sum_mean='sum'):
    '''Returns a trend line using fixed groups'''
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

def trend_line_19(df1, 
                  start_time_str='16:07:24', 
                  end_time_str='16:10:36',
                  sum_mean='sum'
                 ):
    '''Returns a trend line using start and end times'''
    criteria = specific_time_slots(df1, 
                                    start_time_str=start_time_str,
                                    end_time_str=end_time_str,
                                    full_or_filtered_list='filtered')
    if 'temp_criteria_col' not in df1.columns:
        pass
    else:
        df1.drop(['temp_criteria_col'], axis=1)
    df1['temp_criteria_col'] = df1['index'].map(criteria)
    df1['temp_criteria_col'].fillna(0, inplace=True)
    if sum_mean == 'sum':
        temp_df = df1.loc[df1['temp_criteria_col'] == 1]
        temp_df = temp_df.groupby('datetime').sum()
        #temp_df = df1.groupby('temp_criteria_col').sum()
        #temp_df = temp_df.reset_index()
    elif sum_mean == 'mean':
        temp_df = df1.loc[df1['temp_criteria_col'] == 1]
        temp_df = temp_df.groupby('datetime').mean()
        #temp_df = df1.groupby('temp_criteria_col').mean()
        #temp_df = temp_df.reset_index()     
    #temp_df['neg'] = temp_df['neg'].map(lambda x: x * -1) 
    #plt.plot(temp_df['temp_criteria_col'], temp_df['pos'], color='red')
    #plt.plot(temp_df['temp_criteria_col'], temp_df['neg'], color='grey')
    plt.plot(temp_df['pos'], color='red')
    #ax.plot()
    #fig,ax = plt.subplots()
    plt.plot(temp_df['neg'], color='grey')
    plt.xlabel('Seconds')
    plt.ylabel('Sentiment')
    plt.title('Nintendo E3 Twitter Sentiments (sum)')
    plt.legend()
    return plt.show()

def create_dictionary_for_specified_time (df1, 
                                            time=1, 
                                            seconds=5, 
                                            which_five='top'): # choose either 'top' or 'bottom'
    '''Creates a dictionary of words to sum of compound sentiment score'''
    df_filtered_by_seconds = df1.loc[(df1[str(seconds)+'_seconds']== time)]  #| (df_words['five_seconds']== 2)]
    dict_by_seconds = df_filtered_by_seconds.to_dict(orient='index') 
    # create a cleaned dictionary for each word labeled by tweet number
    list_of_word_dicts = []
    for key1, val in dict_by_seconds.items():
        u_words = val['text2'].split(' ')
        neg = val['neg']
        compound = val['compound']
        neu = val['neu']
        pos = val['pos']
        for key, value in val.items():
            try:
                value = float(value)
                if (value > 0) & (key in u_words) :
                    list_of_word_dicts.append({ 
                            'tweet_no': key1,
                            key:{'count': 1, 'compound_sum': compound, 'neg_sum': neg, 
                                 'neu_sum': neu, 'pos_sum': pos},
                                                })
            except:
                pass  
    # remove duplicate words that appear several times in one tweet
    no_dupl_list_of_word_dicts = [i for n, i in enumerate(list_of_word_dicts) 
                                  if i not in list_of_word_dicts[n + 1:]]    
    return_dict = {}
    for i in no_dupl_list_of_word_dicts:
        for key, val in i.items():
            if key is not 'tweet_no':
                if key not in return_dict.keys():
                    return_dict.update({key : val})
                else:
                    return_dict[key]['count'] += val['count']
                    return_dict[key]['compound_sum'] += val['compound_sum']
                    return_dict[key]['neg_sum'] += val['neg_sum']
                    return_dict[key]['neu_sum'] += val['neu_sum']
                    return_dict[key]['pos_sum'] += val['pos_sum']                   
    compound_dict = {}
    for key, val in return_dict.items():
        #print(key, val)
        #compound_dict.update({key: val['compound_sum'] })
        compound_dict[key] = val['compound_sum']
    sorted_compound_dict = sorted(compound_dict.items(), key=lambda kv: kv[1])   
    if which_five == 'top':
        #five_words = dict(sorted_compound_dict[0:5])
        five_words = dict(sorted_compound_dict[-5:])
    elif which_five == 'bottom': 
        #five_words = dict(sorted_compound_dict[-5:])
        five_words = dict(sorted_compound_dict[0:5])
    else:
        "Please choose either 'top' or 'bottom'."
    return five_words

def top_5_dict_to_df(df1, time=1, seconds=5, which_five='top'):  
    '''Identifies the top or bottom 5 words in the dictionary above.'''
    top_5_df = pd.Series(create_dictionary_for_specified_time(df1, 
                                                              time=time, 
                                                              seconds=seconds, 
                                                              which_five=which_five,))
    top_5_df = pd.DataFrame(top_5_df)
    top_5_df = top_5_df.T
    top_5_df['group'] = 'A'
    return top_5_df

def radar_plot_creator(df1, time=1, seconds=5, which_five='top'):
    '''Returns a radar plot using fixed groups'''
    # Set data
    radar_df_test = top_5_dict_to_df(df1,
                                     time=time, 
                                     seconds=seconds, 
                                     which_five=which_five)
    # Make negative values positive
    if which_five=='bottom':
        radar_df_test = radar_df_test.apply(lambda x: x * -1)
    else:
        pass
    # Find largest score, will affect radar plot size
    dictionary = create_dictionary_for_specified_time(df1, time=time, seconds=seconds, which_five=which_five)
    list_of_scores = []
    for k, v in dictionary.items():
        list_of_scores.append(v)
    if which_five=='top':
        relevant_score = max(list_of_scores)
        #relevant_score = relevant_score + (relevant_score*.05)
    elif which_five=='bottom':
        relevant_score = (min(list_of_scores))*-1
        #relevant_score = relevant_score - (.5) 
    # number of variable
    categories=list(radar_df_test)[1:]
    N = len(categories)
    # We are going to plot the first line of the data frame.
    # But we need to repeat the first value to close the circular graph:
    values=radar_df_test.loc[0].drop('group').values.flatten().tolist()
    values += values[:1]
    values
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)
    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='grey', size=8)
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0,(relevant_score*.33),(relevant_score*.66),(relevant_score)], 
               [0, "", "", ""], color="grey", size=7)
    plt.ylim(0,(relevant_score))
    # Plot data
    if which_five == 'top':
        ax.plot(angles, values, linewidth=1, linestyle='solid', color='red')
    elif which_five == 'bottom':
        ax.plot(angles, values, linewidth=1, linestyle='solid', color='grey')
    # Fill area
    if which_five == 'top':
        return ax.fill(angles, values, 'red', alpha=0.1);  
    elif which_five == 'bottom':
        return ax.fill(angles, values, 'grey', alpha=0.1);    

def completed_words_df():
    '''Adds the fixed groups to the words_df'''
    # Load cleaned DF
    with open('cleaned_twitter_df2.pkl', 'rb') as f:
        df = pickle.load(f)
    # Load vader sentiments
    with open('vader_output.pkl', 'rb') as f:
        vader_output = pickle.load(f)
    # Combine cleaned df with vader sentiments and add time column
    df = combine_2_dfs(reset_index(df),json_to_df(vader_output))
    df = add_time_to_df(df)
    # Create list of every unique second in the df
    unique_sec_list = unique_seconds_list(df)
    # Create a new column in the df for each relevant second grouping
    all_relevant_seconds_for_grouping = [5,15,30,60,120,180,300,600,900,
                                     1200,1800,2400,3000,3600]
    for i in all_relevant_seconds_for_grouping:
        df[str(i)+'_seconds'] = df['.time.'].map(seconds_dict(i, unique_sec_list))
    #Append all unique words to the df and assign it to a new words_df
    return words_df(df)

def trend_function(time=30, seconds=5, sum_mean='sum', which_five='top', trend_radar='trend'):
    '''Returns a trend line grouped by every 5 seconds for whole hour.'''
    #Load cleaned DF
    with open('cleaned_twitter_df2.pkl', 'rb') as f:
        df = pickle.load(f)
    # Load vader sentiments
    with open('vader_output.pkl', 'rb') as f:
        vader_output = pickle.load(f)
    # Combine cleaned df with vader sentiments and add time column
    df = combine_2_dfs(reset_index(df),json_to_df(vader_output))
    df = add_time_to_df(df)
    # Create list of every unique second in the df
    unique_sec_list = unique_seconds_list(df)
    # Create a new column in the df for each relevant second grouping
    all_relevant_seconds_for_grouping = [5,15,30,60,120,180,300,600,900,
                                     1200,1800,2400,3000,3600]
    for i in all_relevant_seconds_for_grouping:
        df[str(i)+'_seconds'] = df['.time.'].map(seconds_dict(i, unique_sec_list))
    return trend_line(df, seconds=seconds, sum_mean=sum_mean)

def radar_function(time=30, seconds=5, sum_mean='sum', which_five='top', trend_radar='trend'):
    '''Returns a radar for any specified fixed group.'''
    #Append all unique words to the df and assign it to a new words_df
    words_df = completed_words_df()
    #There are 2 .time. columns, remove both and add one back
    words_df = drop_time_from_df(words_df)
    words_df = add_time_to_df(words_df)
    return radar_plot_creator(words_df, time=time, seconds=seconds, which_five=which_five)

def create_dictionary_for_specified_time_19 (df1, which_five='top'): #time=1, seconds=5, which_five='top'): # choose either 'top' or 'bottom'
    df_filtered_by_temp_col = df1.loc[(df1['temp_criteria_col']== 1)]  #| (df_words['five_seconds']== 2)]
    dict_by_seconds = df_filtered_by_temp_col.to_dict(orient='index') 
    # create a cleaned dictionary for each word labeled by tweet number
    list_of_word_dicts = []
    for key1, val in dict_by_seconds.items():
        u_words = val['text2'].split(' ')
        neg = val['neg']
        compound = val['compound']
        neu = val['neu']
        pos = val['pos']
        for key, value in val.items():
            try:
                value = float(value)
                if (value > 0) & (key in u_words) :
                    list_of_word_dicts.append({ 
                            'tweet_no': key1,
                            key:{'count': 1, 'compound_sum': compound, 'neg_sum': neg, 
                                 'neu_sum': neu, 'pos_sum': pos},
                                                })
            except:
                pass  
    # remove duplicate words that appear several times in one tweet
    no_dupl_list_of_word_dicts = [i for n, i in enumerate(list_of_word_dicts) 
                                  if i not in list_of_word_dicts[n + 1:]]    
    return_dict = {}
    for i in no_dupl_list_of_word_dicts:
        for key, val in i.items():
            if key is not 'tweet_no':
                if key not in return_dict.keys():
                    return_dict.update({key : val})
                else:
                    return_dict[key]['count'] += val['count']
                    return_dict[key]['compound_sum'] += val['compound_sum']
                    return_dict[key]['neg_sum'] += val['neg_sum']
                    return_dict[key]['neu_sum'] += val['neu_sum']
                    return_dict[key]['pos_sum'] += val['pos_sum']                   
    compound_dict = {}
    for key, val in return_dict.items():
        #print(key, val)
        #compound_dict.update({key: val['compound_sum'] })
        compound_dict[key] = val['compound_sum']
    sorted_compound_dict = sorted(compound_dict.items(), key=lambda kv: kv[1])   
    if which_five == 'top':
        #five_words = dict(sorted_compound_dict[0:5])
        five_words = dict(sorted_compound_dict[-5:])
    elif which_five == 'bottom': 
        #five_words = dict(sorted_compound_dict[-5:])
        five_words = dict(sorted_compound_dict[0:5])
    else:
        "Please choose either 'top' or 'bottom'."
    return five_words

def top_5_dict_to_df_19(df1, 
                        #time=1, 
                        #seconds=5, 
                        which_five='top'):  
    top_5_df = pd.Series(create_dictionary_for_specified_time_19(df1, 
                                                              #time=time, 
                                                              #seconds=seconds, 
                                                              which_five=which_five,))
    top_5_df = pd.DataFrame(top_5_df)
    top_5_df = top_5_df.T
    top_5_df['group'] = 'A'
    return top_5_df

def radar_plot_creator_19(df1, time=1, seconds=5, which_five='top'):
   # Set data
    radar_df_test = top_5_dict_to_df_19(df1,
                                     #time=time, 
                                     #seconds=seconds, 
                                     which_five=which_five)
    # Make negative values positive
    if which_five=='bottom':
        radar_df_test = radar_df_test.apply(lambda x: x * -1)
    else:
        pass
    # Find largest score, will affect radar plot size
    dictionary = create_dictionary_for_specified_time_19(df1, 
                                                         #time=time, 
                                                         #seconds=seconds, 
                                                         which_five=which_five)
    list_of_scores = []
    for k, v in dictionary.items():
        list_of_scores.append(v)
    if which_five=='top':
        relevant_score = max(list_of_scores)
        #relevant_score = relevant_score + (relevant_score*.05)
    elif which_five=='bottom':
        relevant_score = (min(list_of_scores))*-1
        #relevant_score = relevant_score - (.5) 
    # number of variable
    categories=list(radar_df_test)[1:]
    N = len(categories)
    # We are going to plot the first line of the data frame.
    # But we need to repeat the first value to close the circular graph:
    values=radar_df_test.loc[0].drop('group').values.flatten().tolist()
    values += values[:1]
    values
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)
    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='grey', size=8)
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0,(relevant_score*.33),(relevant_score*.66),(relevant_score)], 
               [0, "", "", ""], color="grey", size=7)
    plt.ylim(0,(relevant_score))
    # Plot data
    if which_five == 'top':
        ax.plot(angles, values, linewidth=1, linestyle='solid', color='red')
    elif which_five == 'bottom':
        ax.plot(angles, values, linewidth=1, linestyle='solid', color='grey')
    # Fill area
    if which_five == 'top':
        testing_radar = ax.fill(angles, values, 'red', alpha=0.1);  
    elif which_five == 'bottom':
        testing_radar = ax.fill(angles, values, 'grey', alpha=0.1);    
    return testing_radar

def automating_radar_plots(df1, start_time, end_time, which_five='top'):
    criteria = specific_time_slots(df1, 
                                    start_time_str=start_time,
                                    end_time_str=end_time,
                                    full_or_filtered_list='filtered')
    if 'temp_criteria_col' not in df1.columns:
        pass
    else:
        df1.drop(['temp_criteria_col'], axis=1)
    #df = df1.drop(['temp_criteria_col'], axis=1)
    df1['temp_criteria_col'] = df1['index'].map(criteria)
    df1['temp_criteria_col'].fillna(0, inplace=True)
    word_df=words_df(df1)
    return radar_plot_creator_19(word_df, 
                          #time=30, 
                          #seconds=30, 
                          which_five=which_five)

def specific_time_slots(df1,
                        start_time_str='16:07:24', 
                        end_time_str='16:10:46',
                        full_or_filtered_list='filtered'):
    #print(start_time_str, end_time_str)
    every_index = df1['index']
    start_time = datetime.datetime.strptime(start_time_str, "%H:%M:%S")
    start_time = start_time.time()
    end_time = datetime.datetime.strptime(end_time_str, "%H:%M:%S")
    end_time = end_time.time()
    specific_times = (df1['datetime'] > start_time) & (df1['datetime'] <= end_time)
    specific_times_final = []
    for i in specific_times:
        if i == False:
            specific_times_final.append(0)
        elif i == True:
            specific_times_final.append(1)
    times_dict = two_series_to_dict(every_index, specific_times_final)
    times_only_dict = dict((k, v) for k, v in times_dict.items() if v == 1)
    if full_or_filtered_list == 'full':
        return times_dict
    elif full_or_filtered_list == 'filtered':
        return times_only_dict

def two_series_to_dict(s1_keys, s2_values):
    keys = s1_keys
    values = s2_values
    dictionary = dict(zip(keys, values))
    return dictionary

# FROM DATA_CLEANING
# FROM DATA_CLEANING
# FROM DATA_CLEANING
# FROM DATA_CLEANING
# FROM DATA_CLEANING
# FROM DATA_CLEANING
# FROM DATA_CLEANING
# FROM DATA_CLEANING
# FROM DATA_CLEANING

def add_time_from_created(df1):
   df1['.time.'] = df1['created_at'].map(lambda x: x[11:19])

def create_time_col_19(df1): # returns datetime
    add_time_from_created(df1)
    df1['datetime'] = df1['.time.'].map(lambda x: datetime.datetime.strptime(x, "%H:%M:%S"))
    df1['datetime'] = df1['datetime'].map(lambda x: x.time())

def create_timestamp_col(df1): 
    df1['timestamp'] = df1['created_at'].apply(pd.Timestamp)

def select_relevant_cols(df1, col_names=['user.id', 'text', 'lang', 'created_at', 'timestamp_ms']):
    df = df1[col_names]
    return df

def select_relevant_cols_19(df1, col_names=['user.id', 'text', 'lang', 'created_at']):
    df = df1[col_names]
    return df

def filter_lang(df1, lang='en', col_names=['user.id', 'text', 'lang', 'created_at', 'timestamp_ms']):
    df = select_relevant_cols(df1, col_names=col_names)
    df = df.loc[df['lang'] == lang]
    return df

def filter_lang_19(df1, lang='en', col_names=['user.id', 'text', 'lang', 'created_at']):
    df = select_relevant_cols_19(df1, col_names=col_names)
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

# FILES FOR WEB

# def trend_line_for_web(df, 
#                   start_time_str='16:07:24', 
#                   end_time_str='16:10:36',
#                   sum_mean='sum'
#                  ):
#     '''Returns a trend line using start and end times'''
#     criteria = specific_time_slots(df, 
#                                     start_time_str=start_time_str,
#                                     end_time_str=end_time_str,
#                                     full_or_filtered_list='filtered')
#     if 'temp_criteria_col' not in df.columns:
#         pass
#     else:
#         df.drop(['temp_criteria_col'], axis=1)
#     df['temp_criteria_col'] = df['index'].map(criteria)
#     df['temp_criteria_col'].fillna(0, inplace=True)
#     if sum_mean == 'sum':
#         temp_df = df.loc[df['temp_criteria_col'] == 1]
#         temp_df = temp_df.groupby('datetime').sum()
#         #temp_df = df1.groupby('temp_criteria_col').sum()
#         #temp_df = temp_df.reset_index()
#     elif sum_mean == 'mean':
#         temp_df = df.loc[df['temp_criteria_col'] == 1]
#         temp_df = temp_df.groupby('datetime').mean()
#         #temp_df = df1.groupby('temp_criteria_col').mean()
#         #temp_df = temp_df.reset_index()      
#     #plt.plot(temp_df['temp_criteria_col'], temp_df['pos'], color='red')
#     #plt.plot(temp_df['temp_criteria_col'], temp_df['neg'], color='grey')
#     fig,ax = plt.subplots()
#     ax.plot(temp_df['pos'], color='red')
#     ax.plot(temp_df['neg'], color='grey')
#     #ax.xlabel('Seconds')
#     #ax.ylabel('Sentiment')
#     #ax.title('Nintendo E3 Twitter Sentiments (sum)')
#     ax.legend()
#     return fig.savefig('nintendo/webapp/templates/trend_line_output.png')

def trend_line_for_web(
                  start_time_str='16:07:24', 
                  end_time_str='16:10:36',
                  sum_mean='sum'
                 ):
    # load cleaned df
    with open('cleaned_twitter_df2.pkl', 'rb') as f:
        df = pickle.load(f)
    # load vader sentiments df
    with open('vader_output.pkl', 'rb') as f:
        vader_output = pickle.load(f)
    df = combine_2_dfs(reset_index(df),json_to_df(vader_output))   
    df = add_time_to_df(df)
    create_time_col_19(df)
    unique_sec_list = unique_seconds_list(df)
    
    '''Returns a trend line using start and end times'''
    criteria = specific_time_slots(df, 
                                    start_time_str=start_time_str,
                                    end_time_str=end_time_str,
                                    full_or_filtered_list='filtered')
    if 'temp_criteria_col' not in df.columns:
        pass
    else:
        df.drop(['temp_criteria_col'], axis=1)
    df['temp_criteria_col'] = df['index'].map(criteria)
    df['temp_criteria_col'].fillna(0, inplace=True)
    if sum_mean == 'sum':
        temp_df = df.loc[df['temp_criteria_col'] == 1]
        temp_df = temp_df.groupby('datetime').sum()
        #temp_df = df1.groupby('temp_criteria_col').sum()
        #temp_df = temp_df.reset_index()
    elif sum_mean == 'mean':
        temp_df = df.loc[df['temp_criteria_col'] == 1]
        temp_df = temp_df.groupby('datetime').mean()
        #temp_df = df1.groupby('temp_criteria_col').mean()
        #temp_df = temp_df.reset_index()  
    temp_df['neg'] = temp_df['neg'].map(lambda x: x * -1)    
    #plt.plot(temp_df['temp_criteria_col'], temp_df['pos'], color='red')
    #plt.plot(temp_df['temp_criteria_col'], temp_df['neg'], color='grey')
    fig,ax = plt.subplots()
    ax.plot(temp_df['pos'], color='red')
    ax.plot(temp_df['neg'], color='grey')
    #ax.xlabel('Seconds')
    #ax.ylabel('Sentiment')
    #ax.title('Nintendo E3 Twitter Sentiments (sum)')
    ax.legend()
    os.remove('nintendo/webapp/tmp/trend_line_output.png')
    fig.savefig('nintendo/webapp/tmp/trend_line_output.png')
    return 'trend_line_output.png'

def radar_plot_creator_for_web(df, time=1, seconds=5, which_five='top'):
   # Set data
    radar_df_test = top_5_dict_to_df_19(df,
                                     #time=time, 
                                     #seconds=seconds, 
                                     which_five=which_five)
    # Make negative values positive
    if which_five=='bottom':
        radar_df_test = radar_df_test.apply(lambda x: x * -1)
    else:
        pass
    # Find largest score, will affect radar plot size
    dictionary = create_dictionary_for_specified_time_19(df, 
                                                         #time=time, 
                                                         #seconds=seconds, 
                                                         which_five=which_five)
    list_of_scores = []
    for k, v in dictionary.items():
        list_of_scores.append(v)
    if which_five=='top':
        relevant_score = max(list_of_scores)
        #relevant_score = relevant_score + (relevant_score*.05)
    elif which_five=='bottom':
        relevant_score = (min(list_of_scores))*-1
        #relevant_score = relevant_score - (.5) 
    # number of variable
    categories=list(radar_df_test)[1:]
    N = len(categories)
    # We are going to plot the first line of the data frame.
    # But we need to repeat the first value to close the circular graph:
    values=radar_df_test.loc[0].drop('group').values.flatten().tolist()
    values += values[:1]
    values
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)
    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='grey', size=8)
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0,(relevant_score*.33),(relevant_score*.66),(relevant_score)], 
               [0, "", "", ""], color="grey", size=7)
    plt.ylim(0,(relevant_score))
    # Plot data
    if which_five == 'top':
        ax.plot(angles, values, linewidth=1, linestyle='solid', color='red')
    elif which_five == 'bottom':
        ax.plot(angles, values, linewidth=1, linestyle='solid', color='grey')
    # Fill area
    if which_five == 'top':
        testing_radar = ax.fill(angles, values, 'red', alpha=0.1);  
    elif which_five == 'bottom':
        testing_radar = ax.fill(angles, values, 'grey', alpha=0.1);    
    return testing_radar

def automating_radar_plots_for_web(df1, start_time, end_time, which_five='top'):
    criteria = specific_time_slots(df1, 
                                    start_time_str=start_time,
                                    end_time_str=end_time,
                                    full_or_filtered_list='filtered')
    if 'temp_criteria_col' not in df1.columns:
        pass
    else:
        df1.drop(['temp_criteria_col'], axis=1)
    #df = df1.drop(['temp_criteria_col'], axis=1)
    df1['temp_criteria_col'] = df1['index'].map(criteria)
    df1['temp_criteria_col'].fillna(0, inplace=True)
    word_df=words_df(df1)
    return radar_plot_creator_for_web(word_df, 
                          #time=30, 
                          #seconds=30, 
                          which_five=which_five)

