# Nintendo_E3_Sentiment_Analysis

### Business Understanding    
Video game companies have annual announcements that are typically live streamed. In these announcements they will reveal new games, new councils, and any major news events relevant to their company. I wanted to develop a model that could pick up on the positive / negative sentiments of twitter users during the livestream of the conference on June 11th 2019. This work would allow me to understand which announcements users liked / disliked the most.  

### Data Understanding   
I used two sources of data. I initially built the model using 2018 data (found here on kaggle: _____). The second source was from requesting developer access from Twitter to pull from the last 30 days using their API. The issue with this was that I could only pull about 20,000 tweets (when in reality there were probably around 100,000 tweets during the conference with the relevant Nintendo hashtags). I had to be strategic, so I choose 4 specific time intervals for 4 specific games and pulled tweets only during those time slots.  

### Data Preparation   
Since I used an NLP model, the data cleaning focused mainly on the text column (the tweet message itself). I did the following steps to clean the data: 
* The data originally came in a json file with dictionaries in dictionaries in dictionaries. I used json_normalize to get the file into a pandas dataframe and flatten out the embedded dictionaries. 
* Next I selected only the columns that were relevant to my work: 'user.id', 'text', 'lang', 'created_at', 'timestamp_ms'
* I filtered only for tweets that were in english ('en')
* Created a new column ('time') that showed only the H:M:S in a cleaned format. 
* Removed any duplicate rows
* Removed any words starting with either: #, @, http
* Removed punctuation
* Forced all words to be lowercase
* Used lemmatization to get the roots of each word (minimized total unique word count.. ran became run, running became run, etc.) 

### Modeling   
The NLP model I chose was Vader as it is tuned for social media language. For example, words such as 'lol' and 'wtf' are not commonly used outside of social media contexts. Vader includes these commonly used social media words in it's lexicon. The lexicon is essentially a dictionary of words that Vader has assigned a sentiment score to. When using Vader on a tweet, it looks at all of the words in the tweet, for example lets use the sentance: "The game is good, the graphics are nice." 
GOOD and NICE appear in Vaders lexicon and have score of 1.9 and 1.8 respectively. Vader adds all of the sentiments in the tweet together and then standardizes the score to be between -1 and 1. This is the compound score for the tweet. Anything above 0 is postive, anything below is negative. There are some instances where the score will be 0 if Vader did not have any of the words in the tweet in it's lexicon. 

Trend Lines

Radar Plots


### Evaluation  
I will compare the model to the manually classified tweets to see how well it performed. I will use k-fold cross validation. 

### Deployment  
The model will be deployed as a Flask app that will show the video of the conference. Beneath the video will be a blue bar (positive sentiment) and red bar (negative sentiment) that track along with the time of the video to show, in real time, which announcements most people had positive or negative reactions toward. 
*Bar Charts:*    
Beneath the video will be a dashboard that shows a bar chart for male/female sentiments (up bars for positive tweets, down for negative). There will be a similar bar chart but for major age groups. And also a bar chart for the major events that occured in the presentation. There can be a bar chart by country/location too. 
*Drop Down:*     
There will be a drop down for certain major events that occurred throughout the talk. There will be dropdowns for: gender, major age group, and location too. In addition you will have the choice to filter only for tweets with positive sentiments or negative sentiments. 
*Map:*    
There will be a map of the world that shows heat charts for where the majority of tweets about the event originated from. There will be a similar 
*Line Chart:*   
There will be a total event line chart that shows the line that is being shown live to the event. This one will be static, but will have points along it that the user can hover over to see what major events occurred to cause certain spikes in the chart. 

### User Story   
Why does Nintendo want this.
When to announce various reveals.
Analyze tweets occurring during the 2019 Nintendo E3 conference for positive/negative sentiments. Will use clustering for each major event occurring during the conference to see what types of sentiments there are towards the various major events by focusing on relevant topics (fancise names, the presenter, new tech). 

Proposal review guidelines:
https://docs.google.com/document/d/1A9kwRsAcpDdulZSSFOYueV0iJDsCz1eQrba6gJvr4GQ/edit?usp=sharing 

