import pandas as pd
import re
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import emoji
from emoji import UNICODE_EMOJI
import string
import sys
import pickle
import warnings

#datatweet = sys.argv
datatweet = "PTM sangat menyebalkan dan kurang mendidik"
df = pd.DataFrame({'text': [datatweet]})


def is_emoji(s, language="en"):
    return s in UNICODE_EMOJI[language]


def add_space(text):
    return ''.join(' ' + char if is_emoji(char) else char for char in text).strip()


def emojis(text):
    add_space_emoji = add_space(text)
    trans_emoji = emoji.demojize(add_space_emoji, delimiters=("", ""))
    return trans_emoji


def df_transEmoji(x): return emojis(x)


df_translatedEmoji = pd.DataFrame(df.text.apply(df_transEmoji))

punctuations = '''!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~'''


def cleaning(text):
    text = re.sub(
        r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' ', str(text))
    text = re.sub('\[vid]', " ", text)
    text = re.sub('\[vid]', " ", text)
    text = re.sub('@[^\s]+', ' ', text)
    text = re.sub('#[^\s]+', ' ', text)
    text = re.sub('\w*\d\w*', ' ', text)
    text = re.sub(r'([a-z])\1+', r'\1', text)
    text = re.sub(r'\b[a-zA-Z]\b', ' ', text)
    text = re.sub('[\s]+', " ", text)
    text = re.sub(' +', " ", text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = ''.join([c for c in text if ord(c) < 128])
    return text


def df_clean(x): return cleaning(x)


df = pd.DataFrame(df.text.apply(df_clean))


def stemming(text):
    factory_stem = StemmerFactory()
    stemmer = factory_stem.create_stemmer()
    text = stemmer.stem(text)
    return text


def stem(x): return stemming(x)


df = pd.DataFrame(df.text.apply(stem))

warnings.filterwarnings("ignore")
loaded_model = pickle.load(open('fix-svm.pkl', 'rb'))
loaded_feature = pickle.load(open('fix-features.pkl', 'rb'))
loaded_vectorizer = pickle.load(open('fix-vectorizer.pkl', 'rb'))
x_vect = loaded_vectorizer.transform(df.text)
y_pred = loaded_model.predict(x_vect.toarray()[:, loaded_feature])
sentimen = ' '.join([text for text in y_pred])
print(sentimen)
