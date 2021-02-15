from rake_nltk import Rake
import string
import re

def get_keywords(tweets):
    rake = Rake()
    rake.extract_keywords_from_sentences(tweets)
    rake_return = []
    for phrase in rake.get_ranked_phrases():
        if (len(phrase.split()) < 4 and len(phrase.split()) > 1) and (phrase[:2].lower() != 'rt' and 'http' not in phrase and phrase.replace(" ", "").isalpha()):
            rake_return.append(phrase.strip(string.punctuation))
    return rake_return
