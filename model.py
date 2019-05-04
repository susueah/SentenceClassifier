import numpy as np
from keras.layers import Input
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Model
from keras.layers.embeddings import Embedding


def make_embedding(tokenizer, embed_dict):
    word_num = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((word_num, 100))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embed_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            pass
            # print("Word " + word + " not in vocab")
    return embedding_matrix, word_num


class Network(Model):
    def __init__(self, word_num, maxlen, embedding_matrix=None):
        self.maxlen = maxlen
        inp = Input(shape=(self.maxlen,))
        if embedding_matrix is not None:
            net = Embedding(word_num, 100, weights=[embedding_matrix], input_length=self.maxlen, trainable=False)(inp)
        else:
            net = Embedding(word_num, 100, input_length=self.maxlen, trainable=False)(inp)
        net = Bidirectional(LSTM(100, return_sequences=True, dropout=0.50), merge_mode='concat')(net)
        net = TimeDistributed(Dense(100, activation='relu'))(net)
        net = Flatten()(net)
        net = Dense(100, activation='relu')(net)
        output = Dense(1, activation='sigmoid')(net)
        super().__init__(inp, output)
