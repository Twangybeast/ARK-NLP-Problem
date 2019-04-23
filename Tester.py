import csv
import random
import time
import unicodedata

import numpy as np

import ErrorClassifier
from Trainer import tags_to_id, create_replace_nn_model, create_arrange_nn_model

import tensorflow as tf
from tensorflow import keras
#import en_core_web_lg
#nlp = en_core_web_lg.load()

from ErrorClassifier import ERROR_TYPES, tokenize, tokenize_pure_words

word_freqs = {}


def load_word_frequencies():
    with open('learned_frequencies.csv') as file:
        reader = csv.reader(file)
        for row in reader:
            word = row[0]
            freq = int(row[1])
            word_freqs[word] = freq


def eval_largest(n1, n2):
    if n1 == n2:
        return int(random.random() * 2)  # default to random
    elif n1 > n2:
        return 0
    else:
        return 1

def load_arrange_neural_network():
    model = create_arrange_nn_model(None)
    model.load_weights('checkpoints/%s_arrange.ckpt' % FILE_NAME)
    return model

def solve_arrange(tokens1, tokens2, tags1, tags2):

    ids1 = list(map(lambda tag: tags_to_id[tag], tags1))
    ids2 = list(map(lambda tag: tags_to_id[tag], tags2))

    x1 = np.array([ids1])
    x2 = np.array([ids2])

    y1 = arrange_model.predict(x1)
    y2 = arrange_model.predict(x2)

    assert y1.size == 1
    assert y2.size == 1

    return eval_largest(y1.item(), y2.item())


# Addition/Removal are symmetrical operations. In the testing dataset, they should be treated the same
def solve_add(tokens1, tokens2, tags1, tags2):
    return 1 - solve_remove(tokens2, tokens1, tags1, tags2)


def solve_remove(tokens1, tokens2, tags1, tags2):
    # default behavior, return the larger one (removal of tokens from larger one is common)
    return 0 # functionally identical to the commented out version
    # return eval_largest(len(tokens1), len(tokens2))


def solve_typo(tokens1, tokens2, tags1, tags2):
    # Find the delta
    d1, d2 = ErrorClassifier.find_delta_from_tokens(tokens1, tokens2)

    w1 = tokenize_pure_words(str(d1))
    w2 = tokenize_pure_words(str(d2))

    def calc_freq_sum(words):
        total = 0
        for word in words:
            if word in word_freqs:
                total += word_freqs[word]
            elif ErrorClassifier.check_word_list(word):
                total += 1
        # don't bother using an average
        return total

    # Assume the one with higher frequencies is correct
    return eval_largest(calc_freq_sum(w1), calc_freq_sum(w2))


def check_all_has_vectors(doc):
    for d in doc:
        if not d.has_vector:
            return False
    return True


def evaluate_average_delta_similarity(delta, start, end):
    if not delta.has_vector:
        return 0
    total = 0.0
    for t in start:
        total += delta.similarity(t) if t.has_vector else 0
    for t in end:
        total += delta.similarity(t) if t.has_vector else 0
    return total

def load_replace_neural_network():
    model = create_replace_nn_model(None, None)
    model.load_weights('checkpoints/%s_replace.ckpt' % FILE_NAME)
    return model
    # return keras.models.load_model('replace.h5')

def solve_replace(tokens1, tokens2, tags1, tags2):
    # Simplest method, simply compare word similarity vectors
    # Find the delta
    d1, d2, s1, s2 = ErrorClassifier.find_all_delta_from_tokens(tokens1, tokens2)
    tags1 = tags1.split()
    tags2 = tags2.split()

    tag_map = {}
    for i in range(len(tokens1)):
        tag_map[tokens1[i]] = tags1[i]
    for i in range(len(tokens2)):
        tag_map[tokens2[i]] = tags2[i]

    delta1, delta2, start, end = ErrorClassifier.find_all_delta_from_tokens(tokens1, tokens2)

    # no difference in replacement tags, don't use neural net, use similarities of word vectors
    if tag_map[delta1[0]] == tag_map[delta2[0]]:
        return eval_largest(evaluate_average_delta_similarity(delta1[0], start, end),
                            evaluate_average_delta_similarity(delta2[0], start, end))
    else:
        # use the neural network
        # preprocess the data
        ids_d1 = list(map(lambda token: tags_to_id[tag_map[token]], delta1))
        ids_d2 = list(map(lambda token: tags_to_id[tag_map[token]], delta2))
        ids_st = list(map(lambda token: tags_to_id[tag_map[token]], start))  # start ids
        ids_en = list(map(lambda token: tags_to_id[tag_map[token]], end))  # end ids

        if len(ids_st) == 0:
            ids_st = [0]
        if len(ids_en) == 0:
            ids_en = [0]

        input_start = ids_st
        input_en    = ids_en

        input_d1    = [ids_d1[0]]
        input_d2    = [ids_d2[0]]

        x1 = [np.array([input_start]), np.array([input_en]), np.array([input_d1])]
        x2 = [np.array([input_start]), np.array([input_en]), np.array([input_d2])]

        y1 = replace_model.predict(x1)
        y2 = replace_model.predict(x2)

        assert y1.size == 1
        assert y2.size == 1

        return eval_largest(y1.item(), y2.item())




def solve(tokens1, tokens2, error_type, tags1, tags2):
    return globals()['solve_' + error_type.lower()](tokens1, tokens2, tags1, tags2)


prediction_freq = [0, 0]
error_correct = {k: 0 for k in ERROR_TYPES}
error_freq = {k: 0 for k in ERROR_TYPES}

TESTING_RANGE = (900000, 1000000)

FILE_NAME = 'train'

# loads model data
load_word_frequencies()
replace_model = load_replace_neural_network()
arrange_model = load_arrange_neural_network()

with open(FILE_NAME + '.txt', encoding='utf-8') as file, open(FILE_NAME + '.spacy.txt') as file_tags:

    progress = 0
    start_time = time.time()
    words_processed = 0
    for line in file:
        progress += 1
        line_tag = file_tags.readline().strip()
        if not (TESTING_RANGE[0] < progress <= TESTING_RANGE[1]):
            continue

        line = line.strip()
        line = unicodedata.normalize('NFKD', line)
        p1, p2 = line.split('\t')
        tags1, tags2 = line_tag.split('\t')

        error_type = ErrorClassifier.classify_error_labeled(p1, p2)
        tokens1, tokens2 = tokenize(p1, p2)
        answer = solve(tokens1, tokens2, error_type, tags1, tags2)

        prediction_freq[answer] += 1
        if answer == 0:
            error_correct[error_type] += 1
        error_freq[error_type] += 1

        # Display progression in number of samples processed, use random to avoid too many (slow) interactions w/
        # console
        words_processed += len(p1.split()) + len(p2.split())
        if progress % 100 == 0:
            print('\rProgress: [{}] Word Processed: [{}] Words per second: [{}] Lines per second: [{}]'
                  .format(progress, words_processed, \
                          words_processed / (time.time() - start_time), (progress / (time.time() - start_time)))
                  , end='')

print(prediction_freq)
print(error_correct)
print(error_freq)
