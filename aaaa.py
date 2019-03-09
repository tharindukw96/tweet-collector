#To Twitter emotion analyze imports
import tweepy
import re
import calendar
import time
import pandas as pd
import os
import _thread

#Keras should run using backend as theano
os.environ['KERAS_BACKEND'] = 'theano'
from emotion_predictor import EmotionPredictor


#####################Azure DB Connection
import pyodbc 
server = 'tcp:twitter-emotion.database.windows.net' 
database = 'tweet_collection' 
username = 'tharindukw96' 
password = 'cpktnwt@GMA2012' 
cnxn = pyodbc.connect('Driver={ODBC Driver 13 for SQL Server};Server=tcp:twitter-emotion.database.windows.net,1433;Database=tweet_collection;Uid=tharindukw96@twitter-emotion;Pwd=cpktnwt@GMA2012;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;')
cursor = cnxn.cursor()

def remove_punct(text):

    text = re.sub('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+','',text) 
    #remove RT(Retweet) mark from tweets
    text = text.replace("RT","")
    return text

def analyze(keyword):
    #Twitter API keys 
    consumer_key = "mYf39MsctHYfzdqna2kLu28K5"
    consumer_secret = "HexjZTEwS8r8swe40clOrQaISCPN7jzoKVflLvGXqEGRvVpTuh"

    access_token = "1068448554-8K0mSRfBzkAh3mu1K6dPodhEK4d7ncIrWI1y4S8"
    access_token_secret = "0g8PRIseg53l8ISa4p55tFGl98WomOSgnnxT0NuLZOJCy"

    #Authenticate the app using Twitter API key and secret
    auth  = tweepy.OAuthHandler(consumer_key,consumer_secret)
    auth.set_access_token(access_token,access_token_secret)

    api  = tweepy.API(auth)

    #Collect the tweets according to given keyword
    #Collect 300 tweets for analyze(3 api calls)
    public_tweets = api.search(keyword+' -RT',lang='en',tweet_mode='extended',count='100')
    #public_tweets += api.search(keyword,lang='en',count='100')
    #public_tweets += api.search(keyword,lang='en',count='100')

    #Tweets text and Tweet time data collect separately
    tweets = []
    times = []

    #TWeet meta data (create time) collect to bellow list
    tweet_meta = []


    for tweet in public_tweets:
        #Convert time to epoch format
        print(len(tweet.full_text))
        tweets.append(remove_punct(tweet.full_text.replace("'","''")))
        times.append(tweet.created_at.strftime("%m/%d/%Y %H:%M:%S"))
        tweet_meta.append(tweet.id)

    # Pandas presentation options
    pd.options.display.max_colwidth = 150   # show whole tweet's content
    pd.options.display.width = 200          # don't break columns
    # pd.options.display.max_columns = 7      # maximal number of columns


    # Predictor for Ekman's emotions in multiclass setting.
    model = EmotionPredictor(classification='ekman', setting='mc', use_unison_model=True)
    predictions = model.predict_classes(tweets)
    
    result = []
    #Iterate the Emotion result data base
    for index, row in predictions.iterrows():
        result.append([tweet_meta[index],row[0],times[index],row[1]])
    return result



def saveData(cursor,result,cnxn):
    query = 'INSERT INTO  TWEET_INFO  (TweetId,Text,Created_at,Emotion) VALUES '
    for tweet in result:
        query += '(\''+str(tweet[0])+'\',\''+tweet[1]+'\',\''+tweet[2]+'\',\''+tweet[3]+'\'),'
    i = len(query)
    query = query[0:i-1]
    #print(query)
    cursor.execute(query)
    cnxn.commit()

def collectTweets():
    global cursor
    global cnxn
    while True:
        result = analyze('trump')
        saveData(cursor,result,cnxn)
        time.sleep(3)
    


#Run thread
try:
   _thread.start_new_thread( collectTweets, ("Thread-1", 2, ) )
except:
   print("Error: unable to start thread")

while True:
    pass



    