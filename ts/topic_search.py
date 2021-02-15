import os
import tweepy
from tweepy import OAuthHandler


consumer_key = os.environ['CONSUMER_KEY_TWITTER']
consumer_secret = os.environ['CONSUMER_SECRET_TWITTER']
access_token = os.environ['ACCESS_TOKEN_TWITTER']
access_token_secret = os.environ['ACCESS_TOKEN_SECRET_TWITTER']

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)


def get_tweets(query, count=50):
    fetched_tweets = api.search(str(query), count=count, result_type='popular')
    return fetched_tweets


def get_trending():
    woeid = 44418
    trending_tweets = api.trends_place(id = woeid, count=10, exclude='hashtags')
    trends = []
    for trend in trending_tweets[0]['trends']:
        trends.append(trend['name'])
    print(trends)
    return trends
