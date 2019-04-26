import csv
import random
import time
import unicodedata

import numpy as np

import ErrorClassifier
from TokenHelper import tokenize, tokenize_pure_words, find_all_delta_from_tokens, find_delta_from_tokens
from NeuralNetworkHelper import PATH_ARRANGE_CHECKPOINT, FILE_NAME, TESTING_RANGE
from NeuralNetworkHelper import tags_to_id
from NNModels import create_nn_model
from NNTest import PATH_CHECKPOINT1 as PATH_REPLACE_CHECKPOINT

ENABLE_SAVE_OUTPUT = False
PATH_TEST_OUT = 'part1.txt'


def main():
    global word_freqs, replace_model, arrange_model
    prediction_freq = [0, 0]
    error_correct = {k: 0 for k in ErrorClassifier.ERROR_TYPES}
    error_freq = {k: 0 for k in ErrorClassifier.ERROR_TYPES}

    word_freqs = {}
    # loads model data
    load_word_frequencies()
    replace_model = load_replace_neural_network()
    arrange_model = load_arrange_neural_network()

    if ENABLE_SAVE_OUTPUT:
        file_out = open(PATH_TEST_OUT, 'w')
    with open(FILE_NAME + '.txt', encoding='utf-8') as file, open(FILE_NAME + '.spacy.txt') as file_tags:
        progress = 0
        start_time = time.time()
        lines_processed = 0
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

            # Randomly scramble to verify symmetry of the prediction algorithm
            scrambled = random.random() > 0.5
            if scrambled:
                p1, p2 = p2, p1
                tags1, tags2 = tags2, tags1
            error_type = ErrorClassifier.classify_error_labeled(p1, p2)
            tokens1, tokens2 = tokenize(p1, p2)
            answer = solve(tokens1, tokens2, error_type, tags1, tags2)
            if scrambled:
                answer = 1 - answer

            if ENABLE_SAVE_OUTPUT:
                file_out.write('{}\n'.format('A' if answer == 0 else 'B'))

            prediction_freq[answer] += 1
            if answer == 0:
                error_correct[error_type] += 1
            error_freq[error_type] += 1

            # Display progression in number of samples processed, use random to avoid too many (slow) interactions w/
            # console
            words_processed += len(p1.split()) + len(p2.split())
            lines_processed += 1
            if progress % 100 == 0:
                print('\rProgress: [{}] Word Processed: [{}] Words per second: [{}] Lines per second: [{}]'
                      .format(lines_processed, words_processed,
                              words_processed / (time.time() - start_time),
                              (lines_processed / (time.time() - start_time)))
                      , end='')
    if ENABLE_SAVE_OUTPUT:
        file_out.flush()
        file_out.close()
    # Returns the frequency of predicting the first or second one
    print(prediction_freq)
    # Prints the number of lines which predicted the first part as the original (same as the number of correct
    # answers when using train.txt), segmented by the error type
    print(error_correct)
    # Prints the distribution of the types of errors
    print(error_freq)


def load_word_frequencies():
    with open('learned_frequencies.csv') as file:
        reader = csv.reader(file)
        for row in reader:
            word = row[0]
            freq = int(row[1])
            word_freqs[word] = freq


# Returns 0 or 1, corresponding to if the first or second one is the original text, given two numbers representing
# the scores of each part (higher is better). Defaults to random when the scores are equal
def eval_largest(n1, n2):
    if n1 == n2:
        return int(random.random() * 2)  # default to random
    elif n1 > n2:
        return 0
    else:
        return 1


def load_arrange_neural_network():
    model = create_nn_model('arrange')
    model.load_weights(PATH_ARRANGE_CHECKPOINT)
    return model


def solve_arrange(tokens1, tokens2, tags1, tags2):
    tags1 = tags1.split()
    tags2 = tags2.split()

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
    return 0  # functionally identical to the commented out version
    # return eval_largest(len(tokens1), len(tokens2))


def solve_typo(tokens1, tokens2, tags1, tags2):
    # Find the delta
    d1, d2 = find_delta_from_tokens(tokens1, tokens2)

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

    # Assume the one with higher frequencies (more common words) is correct
    return eval_largest(calc_freq_sum(w1), calc_freq_sum(w2))


# Checks if every token in the given iterable of tokens has a vector
def check_all_has_vectors(doc):
    for d in doc:
        if not d.has_vector:
            return False
    return True


# Determine how similar the delta token is to the other tokens, returning the sum of the similarity values
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
    model = create_nn_model('replace1')
    model.load_weights(PATH_REPLACE_CHECKPOINT)
    return model
    # return keras.models.load_model('replace.h5')


def solve_replace(tokens1, tokens2, tags1, tags2):
    # No longer needs the tags
    # Find the delta
    delta1, delta2, start, end = find_all_delta_from_tokens(tokens1, tokens2)

    # use the neural network
    # Convert to word vectors
    start = np.array(list(map(lambda t: t.vector, start)))
    end   = np.array(list(map(lambda t: t.vector, end)))


    assert len(delta1) > 0 and len(delta2) > 0

    delta1 = delta1[0].vector
    delta2 = delta2[0].vector

    vector_length = len(delta1)

    # Ensure non-zero length
    if len(start) == 0:
        start = [[0.] * vector_length]
    if len(end) == 0:
        end = [[0.] * vector_length]

    start = np.reshape(start, (1, len(start), vector_length))
    end   = np.reshape(end, (1, len(end), vector_length))
    delta1 = np.reshape(delta1, (1, vector_length))
    delta2 = np.reshape(delta2, (1, vector_length))

    x = [start, end, delta1, delta2]
    y = replace_model.predict(x, steps=1)

    assert y.size == 1

    # 0 means the first one is better
    return eval_largest(0.5, y.item())


def solve(tokens1, tokens2, error_type, tags1, tags2):
    return globals()['solve_' + error_type.lower()](tokens1, tokens2, tags1, tags2)


if __name__ == '__main__':
    main()
