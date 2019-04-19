import csv
import random
import unicodedata

import ErrorClassifier
from ErrorClassifier import ERROR_TYPES, tokenize, tokenize_pure_words

word_freqs = {}
def load_word_frequencies():
    with open('learned_frequencies.csv') as file:
        reader = csv.reader(file)
        for row in reader:
            word = row[0]
            freq = int(row[1])
            word_freqs[word] = freq

def solve_ARRANGE(p1, p2):
    return int(random.random() * 2) # default to random

# Addition/Removal are symmetrical operations. In the testing dataset, they should be treated the same
def solve_ADD(p1, p2):
    return 1 - solve_REMOVE(p2, p1)

def solve_REMOVE(p1, p2):
    # default behavior, return the larger one (removal of tokens from larger one is common)
    t1, t2 = tokenize(p1, p2)
    if len(t1) > len(t2):
        return 0
    else:
        return 1

def solve_TYPO(p1, p2):
    # Find the delta
    t1, t2 = tokenize(p1, p2)
    d1, d2 = ErrorClassifier.find_delta_from_tokens(t1, t2)

    w1 = tokenize_pure_words(' '.join(d1))
    w2 = tokenize_pure_words(' '.join(d2))

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
    if calc_freq_sum(w1) > calc_freq_sum(w2):
        return 0
    else:
        return 1

def solve_REPLACE(p1, p2):
    return int(random.random() * 2)  # default to random

def solve(p1, p2, error_type):
    return globals()['solve_'+error_type](p1, p2)



prediction_freq = [0, 0]
error_correct = {k: 0 for k in ERROR_TYPES}
error_freq = {k: 0 for k in ERROR_TYPES}

# loads model data
load_word_frequencies()

with open('train.txt', encoding='utf-8') as fin:
    progress = 0
    for line in fin:
        progress += 1
        if progress < 900000:
            continue

        line = line.rstrip()
        line = unicodedata.normalize('NFKD', line)
        p1, p2 = line.split('\t')
        error_type = ErrorClassifier.classify_error_labeled(p1, p2)
        answer = solve(p1, p2, error_type)

        prediction_freq[answer] += 1
        if answer == 0:
            error_correct[error_type] += 1
        error_freq[error_type] += 1

        #Display progression in number of samples processed, use random to avoid too many (slow) interactions w/ console
        if random.random() < 0.001:
            print('\rProgress: [{0}]'.format(progress), end='')

print(prediction_freq)
print(error_correct)
print(error_freq)