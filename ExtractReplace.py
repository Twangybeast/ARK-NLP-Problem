import csv
import os
import random
import time
import unicodedata

from functools import reduce

import tensorflow as tf
from tensorflow import keras

import ErrorClassifier
from NeuralNetworkHelper import TESTING_RANGE, FILE_NAME


# ExtractReplace.py
# Intended to copy over the FILE_NAME .txt to the target file (FILE_NAME .replace.original.txt)
# Only copies lines in which the error type is REPLACE
# Converts the line from "part1, part2" format to "start, end, delta1, delta2" format
# start: the leading tokens that match
# end: the trailing tokens that match
# delta1: the word in part1 that is replaced by delta2 in part2
# delta2: opposite of delta1
# Intended to be used with a neural network later


def main():
    # .spacy.txt is a pre-processed file containing a tokenized
    with open(FILE_NAME + '.txt', encoding='utf-8') as file, open(FILE_NAME + '.spacy.txt') as tags_file:
        progress = 0
        start_time = time.time()
        words_processed = 0
        for line in file:
            line_tag = tags_file.readline().strip()
            progress += 1
            if TESTING_RANGE[0] < progress <= TESTING_RANGE[1]:
                continue

            line = line.strip()
            line = unicodedata.normalize('NFKD', line)

            p1, p2 = line.split('\t')
            t1, t2 = line_tag.split('\t')

            error_type = ErrorClassifier.classify_error_labeled(p1, p2)
            train(p1, p2, error_type, t1, t2)

            # Display progression in number of samples processed, use random to avoid too many (slow)
            # interactions w/ console
            words_processed += len(p1.split()) + len(p2.split())
            if progress % 100 == 0:
                print('\rProgress: [{}] Word Processed: [{}] Words per second: [{}] Lines per second: [{}]'
                      .format(progress, words_processed,
                              words_processed / (time.time() - start_time), (progress / (time.time() - start_time)))
                      , end='')
    train_replace_nn()


def train_replace_nn():
    # saves the data to a file
    assert len(train_delta1) == len(train_delta2) == len(train_start) == len(train_end)
    samples = len(train_delta1)
    with open(FILE_NAME+'.replace.original.txt', 'w', encoding='utf-8') as file_replace:
        file_replace.write('{}\n'.format(samples))
        for i in range(samples):
            file_replace.write(train_start[i] + '\t')
            file_replace.write(train_end[i] + '\t')
            file_replace.write(str(train_delta1[i]) + '\t')
            file_replace.write(str(train_delta2[i]) + '\n')


def prepare_replace_tags(part1, part2, tags1, tags2):
    global test1, test2
    tokens1, tokens2 = ErrorClassifier.tokenize(part1, part2)
    tags1 = tags1.split()
    tags2 = tags2.split()
    assert len(tokens1) == len(tags1)
    assert len(tokens2) == len(tags2)

    delta1, delta2, start, end = ErrorClassifier.find_all_delta_from_tokens(tokens1, tokens2)

    train_start.append(str(start))
    train_end.append(str(end))
    train_delta1.append(str(delta1))
    train_delta2.append(str(delta2))



def train(p1, p2, error_type, t1, t2):
    if error_type == 'REPLACE':
        prepare_replace_tags(p1, p2, t1, t2)


if __name__ == '__main__':

    # For the REPLACE neural network
    train_start = []
    train_end = []
    train_delta1 = []
    train_delta2 = []

    main()
