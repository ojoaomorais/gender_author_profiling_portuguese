import string
import nltk as nltk
import re
from nltk.corpus import brown
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('genesis')
nltk.download('universal_tagset')
from nltk import ngrams
from nltk import word_tokenize, ngrams

def wordCounter(text):
    vector = text.split(" ")
    return len(vector)

def posTagCount(text):
    cleanedText = re.sub("\$.*?\$", '', text)
    table = str.maketrans(dict.fromkeys(string.punctuation))
    cleanedText = cleanedText.translate(table)
    tokens = nltk.tokenize.word_tokenize(cleanedText)
    pos = nltk.pos_tag(tokens)
    i = 0
    tags = []
    for tag in pos:
       tags.append(tag[1])
    return ' '.join(str(e) for e in tags)

def grams(text):
    cleanedText = re.sub("\$.*?\$", '', text)
    table = str.maketrans(dict.fromkeys(string.punctuation))
    cleanedText = cleanedText.translate(table)
    grams = list(ngrams(cleanedText, 3))
    grams = [''.join(tups) for tups in grams]
    return grams