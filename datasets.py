import os

import pandas
import numpy as np

from tokenizer import get_tokens
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

from sklearn import preprocessing


def prepare_ask0729(file_path):
    csv = pandas.read_csv(file_path, '\t')
    # csv = csv.sample(frac=1)
    sents = csv['sentence'].values
    is_intent = [is_in == 'Yes' for is_in in csv['is_intent']]
    return sents, is_intent


class Datasets:
    def __init__(self, data_dir):
        self.dataset1_path = os.path.join(data_dir, "Ask0729-fixed.txt")
        self.dataset2_path = os.path.join(data_dir, "positive_intents.csv")


    def get_data(self):
        datasets_sents = []
        datasets_is_intent = []
        dataset1_sents, dataset1_is_intent = prepare_ask0729(self.dataset1_path)
        datasets_sents.append(dataset1_sents)
        datasets_is_intent.append(dataset1_is_intent)
        dataset2_sents, dataset2_is_intent = self.prepare_positive_intents(self.dataset2_path)
        datasets_sents.append(dataset2_sents)
        datasets_is_intent.append(dataset2_is_intent)
        sents = np.concatenate(datasets_sents)
        is_intent = np.concatenate(datasets_is_intent)
        dual = np.stack((sents, is_intent), axis=1)
        print(dual.shape)
        # print(dual)
        dual = np.random.permutation(dual)
        sents = dual[:, 0]
        is_intent = dual[:, 1]
        print(sents)
        print(np.count_nonzero(is_intent), len(is_intent) - np.count_nonzero(is_intent))
        print(is_intent)
        return sents, is_intent

    def get_tokenized_data(self, max_sentence_len):
        sents, is_intent = self.get_data()
        # token_list = (data['sentence'].apply(get_tokens))
        token_list = [get_tokens(sent) for sent in sents]
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(token_list)
        X, Y = self.get_netio(is_intent, token_list, max_sentence_len, tokenizer)
        return X, Y, tokenizer

    def get_netio(self, is_intent, token_list, max_sentence_len, tokenizer):
        le = preprocessing.LabelEncoder()
        Y = is_intent
        Y = le.fit_transform(Y)

        encoded_text = tokenizer.texts_to_sequences(token_list)
        X = pad_sequences(encoded_text, maxlen=max_sentence_len, padding='post')
        return X, Y

    def prepare_positive_intents(self, file_path):
        csv = pandas.read_csv(file_path, ',', encoding='UTF-8')
        # csv = csv.sample(frac=1)
        sents = csv['sentence'].values
        is_intent = [True for is_in in csv['intent']]
        return sents, is_intent
