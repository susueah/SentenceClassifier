import numpy as np
from keras.layers import Input
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Model
from keras.layers.embeddings import Embedding


def make_embedding(tokenizer, word2vec):
    word_num = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((word_num, 100))
    for word, i in tokenizer.word_index.items():
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            print("Word " + word + " not in vocab")
    return embedding_matrix, word_num


class Network(Model):
    def __init__(self, word_num, maxlen, embedding_matrix=None):
        self.maxlen = maxlen
        inp = Input(shape=(self.maxlen,))
        if embedding_matrix is not None:
            model = Embedding(word_num, 100, weights=[embedding_matrix], input_length=self.maxlen, trainable=False)(inp)
        else:
            model = Embedding(word_num, 100, input_length=self.maxlen, trainable=False)(inp)
        model = Bidirectional(LSTM(100, return_sequences=True, dropout=0.50), merge_mode='concat')(model)
        model = TimeDistributed(Dense(100, activation='relu'))(model)
        model = Flatten()(model)
        model = Dense(100, activation='relu')(model)
        output = Dense(1, activation='sigmoid')(model)
        super().__init__(inp, output)