import pandas as pd
import re
from nltk.tokenize import word_tokenize
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import emoji
from emoji import UNICODE_EMOJI
import string
import sys

df = pd.read_csv('data PTM olah EDA.csv')
X = df.text
y = df.Label

kf = KFold(n_splits=5, random_state=None, shuffle=False)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

vectorizer = TfidfVectorizer(
    use_idf=True, strip_accents="ascii", max_features=1800)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

svm1 = svm.SVC(probability=True, kernel='rbf')
svm1.fit(X_train_vect, y_train)

datatweet = sys.argv
#datatweet = "PTM ribet"
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

x_vect = vectorizer.transform(df.text)
y_pred = svm1.predict(x_vect)
Label = ' '.join([text for text in y_pred])
print(Label)
