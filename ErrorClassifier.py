import pprint
import random
import re
import time
import unicodedata
from enum import Enum
from functools import reduce

import Levenshtein
import spacy
nlp_tokenizer = spacy.load('en_core_web_lg', disable=['tagger', 'parser', 'ner'])


ERROR_TYPES = ['ARRANGE', 'ADD', 'REMOVE', 'TYPO', 'REPLACE']


def tokenize(p1, p2):
    t1 = nlp_tokenizer(p1)
    t2 = nlp_tokenizer(p2)
    return t1, t2

def tokenize_pure_words(part):
    part = re.sub('[^a-zA-Z0-9 ]', ' ', part).lower()
    words = part.split()
    return words


def generate_word_set(tokens):
    s = set()
    for t in tokens:
        i = 0
        while str(t) + str(i) in s:
            i += 1
        s.add(str(t) + str(i))
    return s


def generate_word_dict(tokens):
    d = {}
    for t in tokens:
        if str(t) in d:
            d[str(t)] += 1
        else:
            d[str(t)] = 1
    return d


def dictionary_difference(d1, d2):
    d = {}
    for k in d1:
        v = d1[k]
        v -= d2[k] if k in d2 else 0
        if v > 0:
            d[k] = v
    return d


def freq_to_list(d):
    res = []
    for k in d:
        for i in range(d[k]):
            res.append(k)
    return res

def find_all_delta_from_tokens(t1, t2):
    starting_match = 0
    for i in range(min(len(t1), len(t2))):
        if str(t1[i]) != str(t2[i]):
            starting_match = i
            break

    ending_match = 0
    for i in range(min(len(t1), len(t2))):
        if str(t1[-i - 1]) != str(t2[-i - 1]):
            ending_match = i
            break

    start = t1[:starting_match];
    end = t1[len(t1) - ending_match:]
    return t1[starting_match:len(t1) - ending_match], t2[starting_match:len(t2) - ending_match], start, end


def find_delta_from_tokens(t1, t2):
    delta1, delta2, _, _ = find_all_delta_from_tokens(t1, t2)
    return delta1, delta2


def load_words_list():
    word_files = ['words.txt', 'learned_words.txt']
    res = set()
    for word_file in word_files:
        with open(word_file) as fin:
            res.update(map(lambda x: x.lower(), fin.read().split()))
    return res


def is_num(word):
    try:
        val = int(word)
        return True
    except ValueError:
        return False


def check_word_list(word):
    if is_num(word):
        return True
    if word in words_list:
        return True
    else:
        return False


def all_in_words_list(part):
    words = tokenize_pure_words(part)
    for word in words:
        if check_word_list(word) == False:
            return False
    return True


def classify_error_labeled(part1="", part2="", tokens1 = None, tokens2 = None):
    global words_list
    # replacement, misspelling/space shift, rearrangement, add, remove
    # addition - p1 is a subset of p2
    # removal - p2 is a subset of p1
    # rearrangement - p1 is same set as p2
    # replacement - delta that has large levenshtein
    # mispelling - delta with small levenshtein
    if tokens1 is None or tokens2 is None:
        tokens1 , tokens2 = tokenize(part1, part2)
    else:
        part1 = str(tokens1)
        part2 = str(tokens2)

    dict1 = generate_word_dict(tokens1)
    dict2 = generate_word_dict(tokens2)

    # Inefficient
    # size1 = reduce(lambda total, k: total + dict1[k], dict1, 0)
    # size2 = reduce(lambda total, k: total + dict2[k], dict2, 0)

    size1 = len(tokens1)
    size2 = len(tokens2)

    # Only can have one type of error (supposedly)
    if size1 == size2:
        # possible rearrangement
        res = dictionary_difference(dict1, dict2)
        if len(res) == 0:
            return 'ARRANGE'
    elif size1 < size2:
        # possible addition
        res = dictionary_difference(dict1, dict2)
        if len(res) == 0:
            return 'ADD'
    else:
        # possible removal
        res = dictionary_difference(dict2, dict1)
        if len(res) == 0:
            return 'REMOVE'

    d1, d2 = find_delta_from_tokens(tokens1, tokens2)

    delta1 = str(d1)
    delta2 = str(d2)

    r = Levenshtein.ratio(delta1, delta2)
    d = Levenshtein.distance(delta1, delta2)

    if d <= 2 or r > 0.8:
        # Consider as a typo

        return 'TYPO'
    else:
        # Consider a replacement
        return 'REPLACE'


def generator_file_tokenizer(file_name, use_line_num=lambda n: True, encoding=None):
    def parts_yielder():
        with open(file_name, encoding=encoding) as file:
            progress = 0
            for line in file:
                progress += 1
                if not use_line_num(progress):
                    continue
                line = line.strip()
                line = unicodedata.normalize('NFKD', line)
                p1, p2 = line.split('\t')
                yield p1
                yield p2
    docs = nlp_tokenizer.pipe(parts_yielder(), batch_size=1000)
    iterator = iter(docs)
    while True:
        try:
            tokens1 = next(iterator)
        except StopIteration:
            break
        try:
            tokens2 = next(iterator)
        except StopIteration:
            assert False
        yield tokens1, tokens2


words_list = load_words_list()


if __name__ == '__main__':
    error_frequency = {k: 0 for k in ERROR_TYPES}

    progress = 0
    start_time = time.time()
    words_processed = 0
    for tokens1, tokens2 in generator_file_tokenizer('train.txt', encoding='utf-8'):
        progress += 1
        error = classify_error_labeled(tokens1=tokens1, tokens2=tokens2)
        error_frequency[error] += 1

        # Display progression in number of samples processed, use random to avoid too many (slow) interactions w/
        # console
        words_processed += len(tokens1) + len(tokens2)
        if progress % 100 == 0:
            print('\rProgress: [{}] Word Processed: [{}] Words per second: [{}] Lines per second: [{}]'
                  .format(progress, words_processed, words_processed / (time.time() - start_time),
                          (progress / (time.time() - start_time))), end='')
    print()
    print(error_frequency)

    print('done')
