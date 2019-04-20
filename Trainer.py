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
ENABLE_TRAIN_EMBEDDINGS = True

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


word_to_id = {}
# TODO consider advanced tokenization
def learn_embeddings(part):
    words = tokenize_pure_words(part)
    for word in words:
        pass




def train(p1, p2, error_type):
    # note: learn words is done in the error classifier (classification requires knowing the words)
    learn_word_frequencies(p1)  # only train frequencies on first part, second part is corrupted text
    if ENABLE_LEARN_WORDS:
        learn_words(p1)
    pass


if __name__ == '__main__':
    TESTING_RANGE = (900000, 1000000)
    with open('train.txt', encoding='utf-8') as fin:
        progress = 0
        words_processed = 0
        start_time = time.time()
        parts_list = []
        for line in fin:
            progress += 1
            if TESTING_RANGE[0] < progress <= TESTING_RANGE[1]:
                break

            line = line.strip()
            line = unicodedata.normalize('NFKD', line)
            p1, p2 = line.split('\t')

            words_processed += len(p1.split()) + len(p2.split())

            parts_list.append(p1)
            parts_list.append(p2)

        fin.seek(0)
        print('Time: {0}\nWPS: {1}'.format(time.time() - start_time, words_processed/(time.time()-start_time)))
        nlp_tokens_iter = nlp.pipe(parts_list, batch_size=1000).__iter__()
        print('----')
        print('----')
        print('----')
        start_time = time.time()
        progress = 0
        words_processed = 0
        for x in nlp_tokens_iter:
            progress += 1
            words_processed += len(x)
            if random.random() < 0.01:
                print('\rProgress: {0} Words: {1} WPS: {2}'.format(progress,  words_processed, words_processed / (time.time() - start_time)), end='')

        print('Time: {0}\nWPS: {1}'.format(time.time() - start_time, words_processed/(time.time()-start_time)))



        progress = 0
        for line in fin:
            progress += 1
            if TESTING_RANGE[0] < progress <= TESTING_RANGE[1]:
                break

            line = line.strip()
            line = unicodedata.normalize('NFKD', line)
            p1, p2 = line.split('\t')
            error_type = ErrorClassifier.classify_error_labeled(p1, p2)
            train(p1, p2, error_type)

            # Display progression in number of samples processed, use random to avoid too many (slow) interactions w/
            # console
            if random.random() < 0.001:
                print('\rProgress: [{0}]'.format(progress), end='')

    if ENABLE_LEARN_WORDS:
        save_learned_words()
    else:
        assert len(learned_words) == 0
    save_word_frequencies()
