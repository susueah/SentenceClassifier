import nltk
from nltk.corpus import stopwords
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import wordnet

nltk.download('wordnet')
nltk.download('stopwords')
stopwords = set(stopwords.words('english'))

tweet_tokenizer = TweetTokenizer()


def get_sentences(text):
    spans = PunktSentenceTokenizer().span_tokenize(text)
    result = [(text[start:end], start) for (start, end) in spans]
    return result

def get_word(word):
    lemma = wordnet.morphy(word)
    return word if (lemma is None) else lemma

def get_tokens(sentence):
    tokens = tweet_tokenizer.tokenize(sentence)
    tokens = [get_word(token) for token in tokens if (token not in stopwords and len(token) > 1)]
    return tokens
