import nltk
from nltk.corpus import stopwords
from nltk.tokenize import PunktSentenceTokenizer

nltk.download('wordnet')
nltk.download('stopwords')
stopwords = set(stopwords.words('english'))

from nltk.tokenize import TweetTokenizer
from nltk.corpus import wordnet as wn

tknzr = TweetTokenizer()


def get_sentences(text):
    spans = PunktSentenceTokenizer().span_tokenize(text)
    result = [(text[start:end], start) for (start, end) in spans]
    return result


def get_tokens(sentence):
    #     tokens = nltk.word_tokenize(sentence)  # now using tweet tokenizer
    tokens = tknzr.tokenize(sentence)
    tokens = [token for token in tokens if (token not in stopwords and len(token) > 1)]
    tokens = [get_lemma(token) for token in tokens]
    return tokens


def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
