import csv
import random
import unicodedata

import ErrorClassifier

import en_core_web_sm
nlp = en_core_web_sm.load()

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

def evaluate_average_delta_similarity(delta_tokens, starting_match, ending_match):
    delta = ' '.join(delta_tokens)

    # Tokenize and find word vectors
    s_tokens = nlp(starting_match)
    e_tokens = nlp(ending_match)
    d_tokens = nlp(delta_tokens)

    size_d = len(d_tokens)
    assert size_d > 0

    total = 0.0
    for d in d_tokens:
        for t in s_tokens:
            total += d.similarity(t)
        for t in e_tokens:
            total += d.similarity(t)

    return total / size_d

    '''
    i = 0;
    found_d = False
    while i < len(p):
        p_t = p[i]
        if p_t.text == d[i].text:
            assert not found_d
            found_d = True
            while i < len(p):
                p_t = p[i]
                if p_t.text != d[i].text:
                    i -= 1 # For the subsequent i += 1 in the outer loop
                    break
                i += 1
        i += 1
    '''


def solve_REPLACE(p1, p2):
    # Simplest method, simply compare word similarity vectors
    # Find the delta
    t1, t2 = tokenize(p1, p2)

    d1, d2, s1, s2 = ErrorClassifier.find_all_delta_from_tokens(t1, t2)

    d1 = ' '.join(d1)
    d2 = ' '.join(d2)
    s1 = ' '.join(s1)
    s2 = ' '.join(s2)

    #in_d1 = ErrorClassifier.all_in_words_list(d1)
    #in_d2 = ErrorClassifier.all_in_words_list(d2)

    ##if in_d1 and in_d2:
    #    pass

    # greater similarity is better
    sim1 = evaluate_average_delta_similarity(d1, s1, s2)
    sim2 = evaluate_average_delta_similarity(d2, s1, s2)

    if sim1 > sim2:
        return 0
    else:
        return 1
    #return int(random.random() * 2)  # default to random

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

        line = line.strip()
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