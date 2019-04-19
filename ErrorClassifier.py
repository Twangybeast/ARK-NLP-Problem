import pprint
import random
import re
import time
import unicodedata
from enum import Enum

import Levenshtein


def tokenize(p1, p2):
    t1 = p1.split()
    t2 = p2.split()
    return t1, t2

def tokenize_pure_words(part):
    part = re.sub('[^a-zA-Z0-9 ]', ' ', part).lower()
    words = part.split()
    return words


def generate_word_set(tokens):
    s = set()
    for t in tokens:
        i = 0
        while t + str(i) in s:
            i += 1
        s.add(t + str(i))
    return s


def generate_word_dict(tokens):
    d = {}
    for t in tokens:
        if t in d:
            d[t] += 1
        else:
            d[t] = 1
    return d


def find_word_delta(d1, d2):
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
        if t1[i] != t2[i]:
            starting_match = i
            break

    ending_match = 0
    for i in range(min(len(t1), len(t2))):
        if t1[-i - 1] != t2[-i - 1]:
            ending_match = i
            break

    start = t1[:starting_match];
    end = t1[len(t1) - ending_match:]
    assert start == t2[:starting_match]
    assert end == t2[len(t2) - ending_match:]
    return t1[starting_match:len(t1) - ending_match], t2[starting_match:len(t2) - ending_match], start, end

# TODO Use previous function instead

def find_delta_from_tokens(t1, t2):
    starting_match = 0
    for i in range(min(len(t1), len(t2))):
        if t1[i] != t2[i]:
            starting_match = i
            break

    ending_match = 0
    for i in range(min(len(t1), len(t2))):
        if t1[-i - 1] != t2[-i - 1]:
            ending_match = i
            break

    return t1[starting_match:len(t1) - ending_match], t2[starting_match:len(t2) - ending_match]


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

words_list = load_words_list()
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


unknown_words = 0
min_lev = 90
max_lev_r = 0
magic = 0

def classify_error_labeled(part1, part2):
    global words_list
    # replacement, misspelling/space shift, rearrangement, add, remove
    # addition - p1 is a subset of p2
    # removal - p2 is a subset of p1
    # rearrangement - p1 is same set as p2
    # replacement - delta that has large levenshtein
    # mispelling - delta with small levenshtein
    t1 , t2 = tokenize(part1, part2)

    global magic
    if t1 == t2:
        magic += 1

    set1 = generate_word_set(t1)
    set2 = generate_word_set(t2)

    # Only can have one type of error (supposedly)
    if len(set1) == len(set2):
        # possible rearrangement
        res = set1.difference(set2)
        if len(res) == 0:
            return 'ARRANGE'
    elif len(set1) < len(set2):
        # possible addition
        res = set1.difference(set2)
        if len(res) == 0:
            return 'ADD'
    else:
        # possible removal
        res = set2.difference(set1)
        if len(res) == 0:
            return 'REMOVE'

    # Old code used to explore the data
    # Must identify type of replacement
    # d1 = generate_word_dict(t1)
    # d2 = generate_word_dict(t2)

    # delta1 = {k: d1[k] for k in set(d1) - set(d2)}
    # delta2 = {k: d2[k] for k in set(d2) - set(d1)}

    # delta1 = find_word_delta(d1, d2)
    # delta2 = find_word_delta(d2, d1)

    D1, D2 = find_delta_from_tokens(t1, t2)

    D1 = ' '.join(D1)
    D2 = ' '.join(D2)

    r = Levenshtein.ratio(D1, D2)
    d = Levenshtein.distance(D1, D2)

    global unknown_words, min_lev, max_lev_r
    if d > 2 and (not all_in_words_list(D1) or not all_in_words_list(D2)):
        # print(D1)
        # print(D2)
        # print(Levenshtein.ratio(D1,D2))
        # print(Levenshtein.distance(D1, D2))
        # max_lev_r = max(max_lev_r, r)
        # if  0.7 < r <0.8:
        #    print(D1, D2)
        # min_lev = min(min_lev, d)
        unknown_words += 1
    if d <= 2 or r > 0.8:
        # Consider as a typo

        return 'TYPO'
    else:
        # Consider a replacement
        return 'REPLACE'


ERROR_TYPES = ['ARRANGE', 'ADD', 'REMOVE', 'TYPO', 'REPLACE']

if __name__ == '__main__':
    error_frequency = {k: 0 for k in ERROR_TYPES}

    with open('train.txt', encoding='utf-8') as fin:
        progress = 0
        for line in fin:
            l = line.strip()
            l = unicodedata.normalize('NFKD', l)
            p1, p2 = l.split('\t')
            error = classify_error_labeled(p1, p2)
            error_frequency[error] += 1
            # learn_words(p1) # Now unnecessary, all the words are already learned in the file

            #Display progression in number of samples processed, use random to avoid too many (slow) interactions w/ console
            progress += 1
            if random.random() < 0.001:
                print('\rProgress: [{0}]'.format(progress), end='')
    print()
    print(unknown_words)
    print(error_frequency)
    print(magic)
    # print(min_lev)
    # print(max_lev_r)

    print('done')
