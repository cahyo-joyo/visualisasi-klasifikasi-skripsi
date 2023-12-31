# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gcHCVklQJvP92XNgMWP5tlsWFGT2i3bP
"""

import pandas as pd
import csv
import numpy as np
import emoji
from emoji import UNICODE_EMOJI
from nltk.tokenize import RegexpTokenizer
import re
import nltk
from mtranslate import translate
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pickle
import warnings
import sys
datatweet= sys.argv
datatweet = ' '.join([text for text in datatweet])
data = {'tweet':[datatweet]}
df = pd.DataFrame(data)
def is_emoji(s, language="en"):
    return s in UNICODE_EMOJI[language]

def add_space(text):
    return ''.join(' ' + char if is_emoji(char) else char for char in text).strip()

def emojis(text):
  add_space_emoji = add_space(text)
  trans_emoji = emoji.demojize(add_space_emoji, delimiters=("", ""))
  return trans_emoji
df_transEmoji = lambda x: emojis(x)
df_translatedEmoji = pd.DataFrame(df.tweet.apply(df_transEmoji))
def emoticons(text):
  tokens = RegexpTokenizer('\s+', gaps = True).tokenize(text)
  result = []
  filename = "emoticon.csv"

  for w in tokens:
    with open(filename,encoding='utf-8') as myCSVfile:
      dataFromFile = csv.reader(myCSVfile, delimiter=",")

      for row in dataFromFile:
        if w == row[0].strip():
           w  = row[1]
      result.append(w)

  hasil = ' '.join(result)
  return hasil 
df_transEmot = lambda x: emoticons(x)
df_translatedEmot = pd.DataFrame(df_translatedEmoji.tweet.apply(df_transEmot))
def case_folding(text): 
    text = text.lower()
    return text
df_caseFold = lambda x: case_folding(x)
df_caseFolded = pd.DataFrame(df_translatedEmot.tweet.apply(df_caseFold))
punctuations = '''!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~'''
def cleaning(text): 
    text = re.sub('@[^\s]+', '', text) #menghilangkan username
    text = re.sub('\w*\d\w*', '', text) #menghilangkan angka
    text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text) #menghilangkan url
    text = re.sub(r'(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w\.-]*)*\/?\S', '', text) #menghilangkan url
    text = re.sub('#[^\s]+', '', text) #menghilangkan hashtag
    text  =re.sub(r'\b[a-zA-Z]\b', '', text) #menghilangkan huruf tunggal
    for x in text:
      if x in punctuations:
        text = text.replace(x, " ") #menghilangkan tanda baca
    return text
df_clean = lambda x: cleaning(x)
df_cleaned = pd.DataFrame(df_caseFolded.tweet.apply(df_clean))
def kataSingkat(text):
  tokens = nltk.word_tokenize(text)
  result = []
  filename = "kata singkat.csv"

  for w in tokens:
    with open(filename,encoding='utf-8') as myCSVfile:
      dataFromFile = csv.reader(myCSVfile, delimiter=",")

      for row in dataFromFile:
        if w == row[0].strip():
           w  = row[1]
      result.append(w)

  hasil = ' '.join(result)
  return hasil
df_transSingkat = lambda x: kataSingkat(x)
df_singkat = pd.DataFrame(df_cleaned.tweet.apply(df_transSingkat))
def kataBaku(text):
  tokens = nltk.word_tokenize(text)
  result = []
  filename = "kata baku.csv"

  for w in tokens:
    with open(filename,encoding='utf-8') as myCSVfile:
      dataFromFile = csv.reader(myCSVfile, delimiter=",")

      for row in dataFromFile:
        if w == row[0].strip():
           w  = row[1]
      result.append(w)

  hasil = ' '.join(result)
  return hasil
df_transBaku = lambda x: kataBaku(x)
df_baku = pd.DataFrame(df_singkat.tweet.apply(df_transBaku))
df_translate = lambda x: translate(x, "id", "en")
df_translated = pd.DataFrame(df_baku.tweet.apply(df_translate))
def get_stopword(stopwordsfile):
    stopwords=[]
    file_stopwords = open(stopwordsfile, 'r',encoding='utf-8')
    row = file_stopwords.readline()
    while row:
        word = row.strip()
        stopwords.append(word)
        row = file_stopwords.readline()
    file_stopwords.close()
    return stopwords 
stop_words_indo=get_stopword('stopwordsindo.txt')
def removeSW(text):
    tokens = nltk.word_tokenize(text)
    filtered= []
    for w in tokens:
        if w not in stop_words_indo:
          filtered.append(w)
    hasil = ' '.join(filtered)
    return hasil
df_removingSW = lambda x: removeSW(x)
df_removedSW = pd.DataFrame(df_translated.tweet.apply(df_removingSW))
factory_stem = StemmerFactory()
stemmer = factory_stem.create_stemmer()
df_stemming = lambda x: stemmer.stem(x)
dataprocessed = pd.DataFrame(df_removedSW.tweet.apply(df_stemming))
warnings.filterwarnings("ignore")
loaded_model = pickle.load(open('fix-svm.pkl', 'rb'))
loaded_feature = pickle.load(open('fix-features.pkl', 'rb'))
loaded_vectorizer = pickle.load(open('fix-vectorizer.pkl', 'rb'))
x_vect = loaded_vectorizer.transform(dataprocessed.tweet)
y_pred = loaded_model.predict(x_vect.toarray()[:, loaded_feature])
sentimen = ' '.join([text for text in y_pred])
print(sentimen)