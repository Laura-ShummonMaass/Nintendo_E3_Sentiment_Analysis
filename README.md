# Nintendo_E3_Sentiment_Analysis
Analyze tweets occurring during the 2019 Nintendo E3 conference for positive/negative sentiments. Will use clustering for each major event occurring during the conference to see what types of sentiments there are towards different relevant topics. 

*Business Understanding*   
Video game companies have annual announcements that are typically live streamed. In these announcements they will announce new games, new councils, and any (typically positive) major news relevant to their company. I would like to develop a model that can pick up on the positive / negative sentiments of twitter users during the live stream of the conference on June 11th 2019 in order to understand which announcements users liked / disliked. I’d like to develop a dashboard that analyzes the data to see which demographics liked / disliked certain announcements. 

*Data Understanding*   
The data that I will use is twitter data from the twitter api. It will export in a json format and I will make sure the data includes the following: date/time, message, age, gender, location. I will collect data from 5 minutes prior to 5 minutes post the event. 

*Data Preparation*  
The data should be small enough to store on my machine. I will create a pandas dataframe and filter down the columns to only the ones relevant to my analysis (date/time, message, age, gender, location). I will create a column labeled “sentiment” and for a sample of the data I will manually import the sentiment that I believe the user had in their tweet. This column will be populated with either: “positive”, “neutral”, or “negative”. 

*Modeling*   
I will utilize the code that Taeho and Lee created for their mod 3 project. It is in a flask webpage. 
I will use k means clustering for each major event during the conference. I would like to see what groups of users there are and how they feel towards each announcement. (might need to start out by clustering as a whole in order to find the groups)... or maybe just find groups at each point. Can cluster based on certain positive/negative sentiments toward presenter or the announcement. 

*Evaluation*   
I will compare the model to the manually classified tweets to see how well it performed. I will use k-fold cross validation. 

*Deployment*   
The model will be deployed as a Flask app that will show the video of the conference. Beneath the video will be a blue bar (positive sentiment) and red bar (negative sentiment) that track along with the time of the video to show, in real time, which announcements most people had positive or negative reactions toward. 
Bar Charts:   
Beneath the video will be a dashboard that shows a bar chart for male/female sentiments (up bars for positive tweets, down for negative). There will be a similar bar chart but for major age groups. And also a bar chart for the major events that occured in the presentation. There can be a bar chart by country/location too. 
Drop Down:   
There will be a drop down for certain major events that occurred throughout the talk. There will be dropdowns for: gender, major age group, and location too. In addition you will have the choice to filter only for tweets with positive sentiments or negative sentiments. 
Map:   
There will be a map of the world that shows heat charts for where the majority of tweets about the event originated from. There will be a similar 
Line Chart:   
There will be a total event line chart that shows the line that is being shown live to the event. This one will be static, but will have points along it that the user can hover over to see what major events occurred to cause certain spikes in the chart. 


Proposal review guidelines:
https://docs.google.com/document/d/1A9kwRsAcpDdulZSSFOYueV0iJDsCz1eQrba6gJvr4GQ/edit?usp=sharing 

