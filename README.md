# Nintendo E3 Sentiment Analysis

## Slide Deck:
https://docs.google.com/presentation/d/1NU6hYMc0wPDjx770ZnbZ8ElYbiUtLDSJUczJdegwsNo/edit?usp=sharing

## Nintendo E3 2018 STEPS:
1) Clone this repo to your computer.
2) Data Collection: You can download 2018 twitter data for the hour long announcement from Kaggle here: https://www.kaggle.com/xvivancos/tweets-during-nintendo-e3-2018-conference
3) Move the NintendoTweets.json file into the cloned repo
3) DataFrame Creation: Open the "2018_cleaning_and_trends_radars.ipynb" file and run all of the cells. AFTER YOU RUN THROUGH THE NOTEBOOK the functions at the top of the notebook will be functional. They automate the entire process of creating trend lines and radar graphs. 

## Nintendo E3 2019 STEPS:
1) Recieve a developer account from Twitter from: https://developer.twitter.com/en/apply-for-access.html  
2) Pull the tweets and store them (I used Mongo DB)  
3) Clone this repo to your computer.
4) DataFrame Creation: Open the "2019_cleaning_and_trends_radars.ipynb" file and run all of the cells. AFTER YOU RUN THROUGH THE NOTEBOOK the functions at the top of the notebook will be functional. They automate the entire process of creating trend lines and radar graphs. 

## CRISP-DM:

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

I've created positive and negative trend lines that show, in 5 second groups, both the sum and means of the twitter sentiments. 

I've also created radar plots that show, for any given period of time in the presentation, which words contributed most to either positive or negative sentiments. 


### Evaluation  
For evaluation I have gone through and visually validated tweet sentiment scores. The majority seemed to be appropriately categorized, however I did note a few (understandably) incorrectly rated tweets: "holy shit this is wild wtf" has a score of -0.87, when in reality this sounds more like a positive reaction to a brand new announcement from Nintendo. Also, words that are grossly misspelled are not attributed a sentiment score: "yesssssss" has a score of 0 because vader could not identify it as the word "yes".

### Deployment  
The model is deployed as a Flask app that shows the video of the conference. Beneath the video is the sum trend line showing total positive and negative sentiments at each point in the conference. 

The webpage has 2 inputs: one for the start time of interest and one for end time to analyze in the specific trend line (and eventually possibly radar plots.. currently these take too long to create -- 2 mins). When the users input these values the trend line updates.

### User Story   
The use for this work would be deeper than just the high level NLP sentiment anlysis. If I had more data I would have liked to have done a more detailed analysis into the demographic differences. Does sentiment vary across age, gender, location? I think it would also have been ineresting to incorporate actaul sales into the model. Did higher sentiments result in higher sales? 
For now, the sentiment analysis focuses soley on the words contributing the most to either positive or negative sentiments. I was hoping that certain game names would pop out, or certain aspects of the presentation. This would give Nintendo a more detailed understanding of not only what people thought of the presentation overall, but which aspects they specifically liked/disliked 

Proposal review guidelines:
https://docs.google.com/document/d/1A9kwRsAcpDdulZSSFOYueV0iJDsCz1eQrba6gJvr4GQ/edit?usp=sharing 

