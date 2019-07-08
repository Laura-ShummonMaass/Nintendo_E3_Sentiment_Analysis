def reset_index(df):
    # returns df with index reset

def json_to_df(json_file):
     # returns json in pandas df format

def combine_2_dfs(df1, df2):
     # returns a single df with the 2 dfs combined

def add_time_to_df(df1):
    # FOR 2018 -- uses timestampj_ms -- adds .time. column with H,M,S 

def drop_time_from_df(df1):
    # removes all .time. columns from df

def unique_seconds_list(df):
    # returns list of each unique second in df, uses .time.

def second_groupings(seconds, seconds_list):
    # creates a LIST to be put into dict and mapped back to the df for grouping

def seconds_dict(seconds, seconds_list):
     # turns list of second groupings above into DICT so that it can be mapped back

def unique_words_list(df1):
    # returns list of each unique word used in the text2 column

def vectorize_to_df(df1):
    # returns a df of vader sentiments

def words_df(df1):
     # appends the matrix df of words to the original df
    
def completed_words_df():
     # takes the words_df and appends the relevant desired groupings... 2018 only

def trend_line(df1, seconds=5, sum_mean='sum'):
     # returns a trend line of sentiments for the whole hr, grouped by 5 secs
     # needs a column called '5_seconds' in order to work 

def trend_line_19(df1, 
                  start_time_str='16:07:24', 
                  end_time_str='16:10:36',
                  sum_mean='sum'
                 ):
     # returns a trend line for a specified time range

def create_dictionary_for_specified_time (df1, 
                                        time=1, 
                                        seconds=5, 
                                        which_five='top'): # choose either 'top' or 'bottom'
     # ONLY 2018 -- returns a dict of the words with compound scores

def create_dictionary_for_specified_time_19 (df1, which_five='top'): #time=1, seconds=5, which_five='top'): # choose either 'top' or 'bottom'
     # ONLY 2019 -- returns a dictionary of words to sentiment for words in timeframe
     # must have temp_criteria_col already created

def top_5_dict_to_df(df1, time=1, seconds=5, which_five='top'):  
     # ONLY 2018 -- returns the top 5 or bottom 5 words in the dict from create_dictionary above
     # Returns them in the format of a df to be used for radar plots

def top_5_dict_to_df_19(df1, 
                        #time=1, 
                        #seconds=5, 
                        which_five='top'):  
     # ONLY 2019 -- identifies the top/bottom words in the create_dict
     # returns output in a dataframe

def radar_plot_creator(df1, time=1, seconds=5, which_five='top'):
    # ONLY 2018 -- returns top/bottom 5 words in radar plot

def radar_plot_creator_19(df1, time=1, seconds=5, which_five='top'):
    # ONLY 2019 -- returns radar plots for selected timeframe

def trend_function(time=30, seconds=5, sum_mean='sum', which_five='top', trend_radar='trend'):
     # ONLY 2018 -- returns trend after only basic imports 

def radar_function(time=30, seconds=5, sum_mean='sum', which_five='top', trend_radar='trend'):
    # ONLY 2018 -- after defining df and matrix of words, returns radar plots

def automating_radar_plots(df1, start_time, end_time, which_five='top'):
     # ONLY 2019 -- automates the entire radar process
     # plug in a cleaned df

def specific_time_slots(df1,
                        start_time_str='16:07:24', 
                        end_time_str='16:10:46',
                        full_or_filtered_list='filtered'):
     # returns a dictionary of index to boolean (1 if in time range of interest)

def two_series_to_dict(s1_keys, s2_values):
     # returns a dictionary with any 2 series zipped