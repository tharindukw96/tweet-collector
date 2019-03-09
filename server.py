from flask import Flask,jsonify
import tweepy
import re
import calendar
import time
from nltk.tokenize import TweetTokenizer,word_tokenize
from nltk.corpus import sentiwordnet as sn
import nltk
import matplotlib.pyplot as plt
import pandas as pd
import os
#Keras should run using backend as theano
os.environ['KERAS_BACKEND'] = 'theano'
from emotion_predictor import EmotionPredictor
from flask_cors import CORS, cross_origin

app = Flask(__name__)

#Add CORS headers to the request(Allow cross origin requests)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route("/analyze/<keyword>", methods=['GET'])
@cross_origin()
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
    public_tweets = api.search(keyword,lang='en',count='100')
    public_tweets += api.search(keyword,lang='en',count='100')
    public_tweets += api.search(keyword,lang='en',count='100')

    #Tweets text and Tweet time data collect separately
    tweets = []
    times = []

    #TWeet meta data (create time) collect to bellow list
    tweet_meta = []


    for tweet in public_tweets:
        #Convert time to epoch format
        trim = str(tweet.created_at)
        t  = calendar.timegm(time.strptime(trim,"%Y-%m-%d  %H:%M:%S"))
        tweet_meta.append([tweet.created_at,tweet.text])
        tweets.append(remove_punct(tweet.text))
        times.append(t)
    #create time dataframe
    d = {'Time':times}
    
    #convert time dictionary to dataframe
    df = pd.DataFrame(data=d)

    # Pandas presentation options
    pd.options.display.max_colwidth = 150   # show whole tweet's content
    pd.options.display.width = 200          # don't break columns
    # pd.options.display.max_columns = 7      # maximal number of columns


    # Predictor for Ekman's emotions in multiclass setting.
    model = EmotionPredictor(classification='ekman', setting='mc', use_unison_model=True)
    predictions = model.predict_classes(tweets)
    predictions = pd.concat([predictions, df], axis=1, sort=False)
    #m = predictions.groupby(["Time","Emotion"]).size().reset_index(name='counts')

    
    #Classify the emotion analyze result by emotion categories
    m = predictions.groupby(["Time","Emotion"]).size().unstack(fill_value=0).stack().reset_index(name='counts')
    
    df = m

    #Calculate count for each time
    joy = df.loc[df['Emotion'] == 'Joy'].filter(items=['Time', 'counts'])
    anger = df.loc[df['Emotion'] == 'Anger'].filter(items=['Time', 'counts'])
    fear = df.loc[df['Emotion'] == 'Fear'].filter(items=['Time', 'counts'])
    sadness = df.loc[df['Emotion'] == 'Sadness'].filter(items=['Time', 'counts'])
    surprise = df.loc[df['Emotion'] == 'Surprise'].filter(items=['Time', 'counts'])
    disgust = df.loc[df['Emotion'] == 'Disgust'].filter(items=['Time', 'counts'])
   
    #Send json object with tweet meta data and emotion classification data
    return jsonify({"meta":tweet_meta,"joy":joy.to_json(orient='values'),"anger":anger.to_json(orient='values'),"fear":fear.to_json(orient='values'),"sadness":sadness.to_json(orient='values'),"surprise":surprise.to_json(orient='values'),"disgust":disgust.to_json(orient='values')})

def remove_punct(text):

    text = re.sub('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+','',text) 
    #remove RT(Retweet) mark from tweets
    text = text.replace("RT","")
    return text

def tokenize(text):
    #Tokenize the tweet content
    tknzr = TweetTokenizer()
    tokenz = tknzr.tokenize(text)
    return tokenz

def posTagging(tokenz):
    #Add Part Of Speach TAg for every word
    return nltk.pos_tag(tokenz)

def knowledgeBaseValidation(text):
    classArr= []
    for word in text:
        syns = sn.senti_synsets(word)
        pos = 0
        neg = 0
        for j in syns:
            pos += j.pos_score()
            neg+= j.neg_score()
            break
        if(pos == 0):
            if(neg < -0.1 and neg > -0.5 ):
                classArr.append(2)
            elif(neg >= -1 and neg <= -0.5):
                classArr.append(3)
        else:
            if(pos > 0.1 and pos < 0.5 ):
                classArr.append(1)
            elif(pos >= 0.5 and pos <= 1):
                classArr.append(0)
    if(len(classArr)==0):
        return "null"
    else:
        return max(set(classArr),key=classArr.count)

if __name__ == '__main__':
    app.run(debug=True)