import string
import nltk as nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def wordCounter(text):
    vector = text.split(" ")
    return len(vector)

def ponctuationNumber(text):
    tokens = nltk.tokenize.word_tokenize(text)
    pos = nltk.pos_tag(tokens)
    i = 0
    for tag in pos:
        if tag[1] in ["JJ","JJR","JJS"]:
            i = i + 1
    return i