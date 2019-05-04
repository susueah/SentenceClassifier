import argparse

import numpy as np
from keras.callbacks import ModelCheckpoint
from tokenizer import get_tokens
from model import Network
from model import make_embedding
from datasets import Datasets
import pickle
import keras.backend as K

import os
from keras_preprocessing.text import Tokenizer

root_dir = 'C:\\Study\\SentenceClassifier\\resources'
embedding_path = os.path.join(root_dir, "glove.twitter.27B.100d.txt")

max_sentence_len = 50


def get_embed_dict(dict_path):
    with open(dict_path, "r", encoding='UTF-8') as file:
        if file:
            embed_dict = dict()
            for line in file.read().splitlines():
                key = line.split(' ', 1)[0]
                value = np.array([float(val) for val in line.split(' ')[1:]])
                embed_dict[key] = value
            return embed_dict
        else:
            print("invalid file path")


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='LSTM for sentence binary classification')
    parser.add_argument('--weights', dest='weights', default=None,
                        help='initialize with pretrained model weights',
                        type=str)
    args = parser.parse_args()

    return args


def load_model_and_tokenizer(weights_path, maxlen):
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    word_num = len(tokenizer.word_index) + 1
    model = Network(word_num, maxlen=maxlen)
    model.load_weights(weights_path)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model, tokenizer

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

if __name__ == '__main__':
    args = parse_args()
    datasets = Datasets(root_dir)

    X, Y, _ = datasets.get_tokenized_data(max_sentence_len=max_sentence_len)
    model, tokenizer = load_model_and_tokenizer(args.weights, max_sentence_len)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1_m, precision_m, recall_m])
    loss, accuracy, f1, prec, rec = model.evaluate(X[-1000:], Y[-1000:], verbose=1)
    print(accuracy, f1, prec, rec)
    # model.fit(X, Y, validation_split=0.25, epochs=50, verbose=2, callbacks=callbacks_list)
