import nltk
from nltk.corpus import twitter_samples
from nltk.tag import pos_tag
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from  nltk.stem import SnowballStemmer
from nltk import classify
from nltk import NaiveBayesClassifier
import pandas as pd
import re, string

DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"

df = pd.read_csv('training_tweets/training_tweets.csv', encoding =DATASET_ENCODING , names=DATASET_COLUMNS)

decode_map = {0: 'NEGATIVE', 2: 'NEUTRAL', 4: 'POSITIVE'}
def decode_sentiment(label):
    return decode_map[int(label)]
df.target = df.target.apply(lambda x: decode_sentiment(x))

stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")

def preprocess(text, stem=False):
    # Remove link,user and special characters
    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', str(text).lower()).strip()
    token = re.sub("(@[A-Za-z0-9_]+)","", str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)

df.text = df.text.apply(lambda x: preprocess(x))

def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens

pos_df = []
neg_df = []
pos_df = df[df['target'] == 'POSITIVE']
neg_df = df[df['target'] == 'NEGATIVE']
print(len(pos_df))
print(len(neg_df))

pos_tok = []
for i in range(0, len(pos_df)):
    pos_tok.append(remove_noise(word_tokenize(pos_df['text'].iloc[i]), stop_words))
neg_tok = []
for i in range(0, len(neg_tok)):
    neg_tok.append(remove_noise(word_tokenize(neg_df['text'].iloc[i]), stop_words))

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

positive_tokens_for_model = get_tweets_for_model(pos_tok)
negative_tokens_for_model = get_tweets_for_model(neg_tok)

# shuffle the df
df = df.sample(frac=1)
# 80/20 split
train_data = []
test_data = []
train_data = df[['target', 'text']][:1280000]
test_data = df[['target', 'text']][1280000:]

classifier = NaiveBayesClassifier.train(train_data)
print("Accuracy is:", classify.accuracy(classifier, test_data))
print(classifier.show_most_informative_features(10))

import pickle
f = open('classifier.pickle', 'wb')
pickle.dump(classifier, f)
f.close()
