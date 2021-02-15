from flask import Flask, request, render_template
from ts.classifier import clean_tweet
from ts.topic_search import get_tweets, get_trending
from ts.keyword_extract import get_keywords
from ts.model import load_model


app = Flask(__name__)


@app.route('/')
def home():
    trending = get_trending()
    return render_template('index.html', trending=trending)


@app.route('/topic', methods=['POST'])
def topic():
    '''
    Give a subject and get the # of positive and negative tweets
    '''
    pos = 0
    neg = 0
    result = dict()
    tweets_to_rake = []
    tweets = []
    if request.method == 'POST':
        classifier = load_model()
        req = request.json
        subject = request.form['input']
        fetched_tweets = get_tweets(subject, count=200)
        for i in range(0, len(fetched_tweets)):
            tweet = clean_tweet(fetched_tweets[i].text)
            if tweet not in tweets:
                tweets.append(tweet)
        for tweet in tweets:
            sentiment = classifier.classify(dict([token, True] for token in tweet))
            tweets_to_rake.append(' '.join(tweet))
            if sentiment == 'Positive':
                pos = pos + 1
            elif sentiment == 'Negative':
                neg = neg + 1
        keywords = get_keywords(tweets_to_rake)
        result = {'Positive Tweets':pos, 'Negative Tweets':neg, 'Keywords': keywords}
        print(result)
        data = [pos, neg]
    return render_template('result.html', keywords=keywords, data=data)


@app.route('/predict', methods=['POST'])
def predict():
    '''
    Give some tweet text and 'positive' or 'negative' sentiment will be returned
    '''
    if request.method == 'POST':
        classifier = load_model()
        req = request.json
        tweet = req['tweet']
        tweet = clean_tweet(tweet)
        sentiment = classifier.classify(dict([token, True] for token in tweet))
    return sentiment

if __name__ == '__main__':
	app.run(debug=True)
