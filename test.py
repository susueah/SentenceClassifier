import argparse
import json
import pickle

from keras_preprocessing.sequence import pad_sequences

from tokenizer import get_sentences, get_tokens
from model import Network

max_sentence_len = 50
detect_thresh = 0.5


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='LSTM for sentence binary classification')
    parser.add_argument('--weights', dest='weights',
                        help='initialize with pretrained model weights',
                        type=str)
    parser.add_argument('--text', dest='text_path',
                        help='text to run on',
                        type=str)
    parser.add_argument('--out_json', dest='json_path',
                        help='result json',
                        type=str)
    args = parser.parse_args()

    return args


def run_on_input(filename, model, tknzr, maxlen, json_path):
    with open(filename, 'r', encoding='UTF-8') as handle:
        input = json.load(handle)
    output = []
    for text in input:
        id = text['id']
        str = text['text']
        proc_text = {'id': id, 'intents': []}
        sentences_starts = get_sentences(str)
        for sentence, start in sentences_starts:
            is_intent = run_single_sentence(sentence, model, tknzr, maxlen)
            if is_intent:
                proc_text['intents'].append({'text': sentence, 'startsAt': start})
        output.append(proc_text)
    with open(json_path, 'w', encoding='UTF-8') as handle:
        json.dump(output, handle, indent=4)


def run_single_sentence(sentence, model, tknzr, maxlen):
    token_list = [get_tokens(sentence)]
    encoded_text = tknzr.texts_to_sequences(token_list)
    X = pad_sequences(encoded_text, maxlen=maxlen, padding='post')
    Y = model.predict(X)
    # print(Y)
    return Y[0][0] >= detect_thresh


def load_model_and_tokenizer(weights_path, maxlen):
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    word_num = len(tokenizer.word_index) + 1
    model = Network(word_num, maxlen=maxlen)
    model.load_weights(weights_path)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model, tokenizer


if __name__ == '__main__':
    args = parse_args()
    model, tknzr = load_model_and_tokenizer(args.weights, max_sentence_len)
    run_on_input(args.text_path, model, tknzr, max_sentence_len, args.json_path)
