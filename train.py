import argparse

import numpy as np
from keras.callbacks import ModelCheckpoint
from tokenizer import get_tokens
from model import Network
from model import make_embedding
from datasets import Datasets
import pickle

import os
from keras_preprocessing.text import Tokenizer


root_dir = 'C:\\Study\\SentenceClassifier\\resources'
embedding_path = os.path.join(root_dir, "glove.twitter.27B.100d.txt")

max_sentence_len = 50

def get_word2vec(dict_path):
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



if __name__ == '__main__':
    args = parse_args()
    word2vec = get_word2vec(embedding_path)
    # print(dataset1.columns)

    datasets = Datasets(root_dir)

    X, Y, tokenizer = datasets.get_tokenized_data(max_sentence_len=max_sentence_len)
    embedding_matrix, word_num = make_embedding(tokenizer, word2vec)

    model = Network(word_num=word_num, embedding_matrix=embedding_matrix, maxlen=max_sentence_len)
    # if args.weights is not None:
    #     model.load_weights(args.weights)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', ])

    filepath="weights\weights2-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    model.fit(X, Y, validation_split=0.25, epochs=50, verbose=2, callbacks=callbacks_list)
