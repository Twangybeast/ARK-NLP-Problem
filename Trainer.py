import csv
import os
import random
import time
import unicodedata

import en_core_web_lg
nlp = en_core_web_lg.load()

import ErrorClassifier
from ErrorClassifier import ERROR_TYPES, tokenize_pure_words

# Only can learn when the learned_words.txt file is empty
ENABLE_LEARN_WORDS = False
ENABLE_LEARN_EMBEDDING = True

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


def load_tags_to_id():
    tags_to_id = {}
    with open('spacy_tags.txt') as tags_list:
        id = 1
        for line in tags_list:
            tag = line.strip()
            tags_to_id[tag] = id
            id += 1
    return tags_to_id

tags_to_id = load_tags_to_id()

def prepare_tags_for_nn(p1, p2, t1, t2):

    if len() == 0:
        print(t1)
        print(t2)
        print(p1)
        print(p2)
    id1 = tags_to_id[d1[0]]
    id2 = tags_to_id[d2[0]]

    ids1 = map(lambda tag: tags_to_id[tag], t1.split())
    ids2 = map(lambda tag: tags_to_id[tag], t2.split())




def train(p1, p2, error_type, t1, t2):
    # note: learn words is done in the error classifier (classification requires knowing the words)
    learn_word_frequencies(p1)  # only train frequencies on first part, second part is corrupted text
    if ENABLE_LEARN_WORDS:
        learn_words(p1)
    if error_type == 'REPLACE':
        prepare_tags_for_nn(p1, p2, t1, t2)


'''
# creates an index
training_words = []
VOCAB_SIZE = 20000
def learn_embedding(part):
    training_words.append(part)
'''



if __name__ == '__main__':
    TESTING_RANGE = (900000, 1000000)
    # .spacy.txt is a pre-processed file containing a tokenized
    with open('train.txt', encoding='utf-8') as file, open('train.spacy.txt') as ftags:
        test = {}
        progress = 0
        for line in file:
            line_tag = ftags.readline().strip()
            progress += 1
            if TESTING_RANGE[0] < progress <= TESTING_RANGE[1]:
                break

            line = line.strip()
            line = unicodedata.normalize('NFKD', line)

            p1, p2 = line.split('\t')
            t1, t2 = line_tag.split('\t')

            error_type = ErrorClassifier.classify_error_labeled(p1, p2)
            train(p1, p2, error_type, t1, t2)


            if error_type == 'REPLACE':
                d1, d2 = ErrorClassifier.find_delta_from_tokens(t1.split(), t2.split())
                length = max(len(d1), len(d2))
                if length not in test:
                    test[length] = 1
                else:
                    test[length] += 1

            # Display progression in number of samples processed, use random to avoid too many (slow) interactions w/
            # console
            if random.random() < 0.001:
                print('\rProgress: [{0}]'.format(progress), end='')
        print(test)
    if ENABLE_LEARN_WORDS:
        save_learned_words()
    else:
        assert len(learned_words) == 0
    save_word_frequencies()
