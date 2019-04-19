import csv
import os
import random
import unicodedata

import ErrorClassifier
from ErrorClassifier import ERROR_TYPES, tokenize_pure_words


# Only can learn when the learned_words.txt file is empty
LEARN_WORDS_ENABLED = True

learned_words = set()
def learn_words(part):
    words = tokenize_pure_words(part)
    for word in words:
        if not ErrorClassifier.check_word_list(word):
            learned_words.add(word)

def save_learned_words():
    with open('learned_words.txt', 'w') as fout:
        for word in learned_words:
            fout.write(word + '\n')

word_freqs = {}
def learn_word_frequencies(part):
    words = tokenize_pure_words(part)
    for word in words:
        if word in word_freqs:
            word_freqs[word] += 1
        else:
            word_freqs[word] = 1

def save_word_frequencies():
    with open('learned_frequencies.csv', 'w', newline='') as fout:
        csv_writer = csv.writer(fout)
        for word, freq in word_freqs.items():
            # don't bother with the little words
            if freq > 1:
                csv_writer.writerow([word, freq])



def train(p1, p2, error_type):
    # note: learn words is done in the error classifier (classification requires knowing the words)
    learn_word_frequencies(p1) # only train frequencies on first part, second part is corrupted text
    if LEARN_WORDS_ENABLED:
        learn_words(p1)
    pass


with open('train.txt', encoding='utf-8') as fin:
    progress = 0
    for line in fin:
        progress += 1
        if progress > 900000:
            break

        line = line.strip()
        line = unicodedata.normalize('NFKD', line)
        p1, p2 = line.split('\t')
        error_type = ErrorClassifier.classify_error_labeled(p1, p2)
        train(p1, p2, error_type)

        #Display progression in number of samples processed, use random to avoid too many (slow) interactions w/ console
        if random.random() < 0.001:
            print('\rProgress: [{0}]'.format(progress), end='')

if LEARN_WORDS_ENABLED:
    save_learned_words()
else:
    assert len(learned_words) == 0
save_word_frequencies()