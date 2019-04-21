import csv
import random
import time
import unicodedata

import ErrorClassifier

import en_core_web_lg
nlp = en_core_web_lg.load()

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


def solve_arrange(tokens1, tokens2):
    return int(random.random() * 2)  # default to random


# Addition/Removal are symmetrical operations. In the testing dataset, they should be treated the same
def solve_add(tokens1, tokens2):
    return 1 - solve_remove(tokens2, tokens1)


def solve_remove(tokens1, tokens2):
    # default behavior, return the larger one (removal of tokens from larger one is common)
    return 0 # functionally identical to the commented out version
    # return eval_largest(len(tokens1), len(tokens2))


def solve_typo(tokens1, tokens2):
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
        if d.has_vector:
            for t in s_tokens:
                total += d.similarity(t) if t.has_vector else 0
            for t in e_tokens:
                total += d.similarity(t) if t.has_vector else 0

    return total / size_d, check_all_has_vectors(d_tokens)

asdf = [[0, 0, 0, 0], [0,0,0,0]]
def solve_replace(tokens1, tokens2):
    # Simplest method, simply compare word similarity vectors
    # Find the delta
    d1, d2, s1, s2 = ErrorClassifier.find_all_delta_from_tokens(tokens1, tokens2)

    # d1 = ' '.join(d1)
    # d2 = ' '.join(d2)
    # s1 = ' '.join(s1)
    # s2 = ' '.join(s2)

    # in_d1 = ErrorClassifier.all_in_words_list(d1)
    # in_d2 = ErrorClassifier.all_in_words_list(d2)

    ##if in_d1 and in_d2:
    #    pass

    # greater similarity is better
    # sim1, c1 = evaluate_average_delta_similarity(d1, s1, s2)
    # sim2, c2 = evaluate_average_delta_similarity(d2, s1, s2)
    sim1, c1 = 0, 0
    sim2, c2 = 0, 0

    res = eval_largest(sim1, sim2)
    asdf[res][(2 if c1 else 0) + (1 if c2 else 0)] += 1
    return res


def solve(tokens1, tokens2, error_type):
    return globals()['solve_' + error_type.lower()](tokens1, tokens2)


prediction_freq = [0, 0]
error_correct = {k: 0 for k in ERROR_TYPES}
error_freq = {k: 0 for k in ERROR_TYPES}

TESTING_RANGE = (900000, 1000000)

# loads model data
load_word_frequencies()

with open('train.txt', encoding='utf-8') as fin:
    progress = 0
    for line in fin:
        progress += 1
        if not (TESTING_RANGE[0] < progress <= TESTING_RANGE[1]):
            continue

    fin.seek(0)

    progress = 0
    start_time = time.time()
    words_processed = 0
    for line in fin:
        progress += 1
        if not (TESTING_RANGE[0] < progress <= TESTING_RANGE[1]):
            continue

        line = line.strip()
        line = unicodedata.normalize('NFKD', line)
        p1, p2 = line.split('\t')
        error_type = ErrorClassifier.classify_error_labeled(p1, p2)
        tokens1, tokens2 = tokenize(p1, p2)
        answer = solve(tokens1, tokens2, error_type)

        prediction_freq[answer] += 1
        if answer == 0:
            error_correct[error_type] += 1
        error_freq[error_type] += 1

        # Display progression in number of samples processed, use random to avoid too many (slow) interactions w/
        # console
        words_processed += len(p1.split()) + len(p2.split())
        if random.random() < 0.01:
            print('\rProgress: [{0}] Word Processed: [{1}] Words per second: [{2}]'
                  .format(progress, words_processed, (words_processed / (time.time() - start_time))), end='')

print(prediction_freq)
print(error_correct)
print(error_freq)
print(asdf)
