import requests
import json
import pymongo
import time

from pymongo import MongoClient 
#mc = mongo

conn = MongoClient(port=47017)

mc = conn
stored_responses = mc['cache']['responses']

# database 
db = conn['nintendoe3'] 

# Created or Switched to collection names: x_collection
tweet_coll = db['tweets']

def store_tweets_in_mongo(tweets, coll=tweet_coll):
    for tweet in tweets:
        #print(tweet['id'])
        existing_tweets = coll.find({'id': tweet['id']})  #searches to see if the id already exists in the dataframe
        if len(list(existing_tweets))==0:
            coll.insert_one(tweet)

def get_token():
    with open('.secret_token') as f:
        data = json.load(f)
    return data['access_token']

def get_tweets(query='(#NintendoE3 OR #NintendoDirect) lang:en', 
               max_results=100,
               from_date='201906111600',
               to_date='201906111643',
               resp=None,
               token=None,
               url='https://api.twitter.com/1.1/tweets/search/30day/dev.json',
              ):
    """fetches one batch of 100 tweets"""
    if token is None:
        token = get_token()
    if resp is not None:
        resp = json.loads(resp)  #convert bytes to dict
    headers={
        'authorization': f'Bearer {token}',
        'content-type': 'application/json',
    }
    data={'query': query,
         'maxResults': max_results,
         'fromDate': from_date,
         'toDate': to_date,
         }
    if (resp is not None) and ('next' in resp):
        data['next'] = resp['next']
    response=fetch_from_cache_or_api(
        url, 
        headers=headers,
        data=json.dumps(data))
    return response



def fetch_from_cache_or_api(url, headers, data, stored_responses=stored_responses):
    """Caches post requests and retrieves locally if saved."""
    found_responses = stored_responses.find({
        'url': url,
        'post_data': data,
    })
    found_responses = list(found_responses)
    if len(found_responses) > 0:
        print("Found in cache! Yay!")  # delete this
        found_response = found_responses[0]
        return found_response['response']
    else:
        print("Didn't find it. Querying API...")  # delete this?
        response = requests.post(
            url, 
            headers=headers,
            data=data)
        if response.status_code == 200:
            print("Success! Storing...")  # delete this?
            stored_responses.insert_one({
                'url': url,
                'post_data': data,
                'response': response.content
            })  # store int
            return response.content
        else:
            raise Exception("Request Failed:\n" + str(response.content))

def store_response_tweets(response_data):
    response_dict = json.loads(response_data)
    tweets = response_dict['results']
    store_tweets_in_mongo(tweets)

def store_many_tweets( 
    query='(#NintendoE3 OR #NintendoDirect) lang:en', 
    max_results=100,
    from_date='201906111600',
    to_date='201906111643',
    resp=None,
    token=None,
    url='https://api.twitter.com/1.1/tweets/search/30day/dev.json',
    limit=3 #testing to make sure it's working
    ): 
    
    i = 0
    
    while (resp is None) or ('next' in json.loads(resp)):
        resp = get_tweets(
            query = query,
            max_results=max_results,
            from_date=from_date,
            to_date=to_date,
            resp=resp,
            token=token,
            url=url,
                  )
        store_response_tweets(resp)
        
        i += 1
        print(i)
        if i > limit:
            break
        time.sleep(2)