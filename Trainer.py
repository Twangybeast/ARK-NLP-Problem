import csv
import os
import random
import time
import unicodedata

# import en_core_web_lg
# nlp = en_core_web_lg.load()
from functools import reduce

import tensorflow as tf
from tensorflow import keras
#tf.enable_eager_execution()

import ErrorClassifier
from ErrorClassifier import ERROR_TYPES, tokenize_pure_words

# Only can learn when the learned_words.txt file is empty
ENABLE_LEARN_WORDS = False
ENABLE_LEARN_EMBEDDING = True
ENABLE_TRAIN_REPLACE_NN = True

TRAIN_FROM_DISK_REPLACE = True

ENABLE_TRAIN_ARRANGE_NN = False  # True when we want to train the ARRANGE neural network
ENABLE_PROCESS_ARRANGE_DATA = True # True when we want to process the original .txt file for the dataset

ONLY_TRAIN_NN = True

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

# For the REPLACE neural network
train_start = []
train_end = []
train_delta1 = []
train_delta2 = []

test1 = 0
test2= 0
def prepare_tags_for_nn(part1, part2, tags1, tags2):
    global test1, test2
    tokens1, tokens2 = ErrorClassifier.tokenize(part1, part2)
    tags1 = tags1.split()
    tags2 = tags2.split()
    assert len(tokens1) == len(tags1)
    assert len(tokens2) == len(tags2)

    tag_map = {}
    for i in range(len(tokens1)):
        tag_map[tokens1[i]] = tags1[i]
    for i in range(len(tokens2)):
        tag_map[tokens2[i]] = tags2[i]

    delta1, delta2, start, end = ErrorClassifier.find_all_delta_from_tokens(tokens1, tokens2)

    ids_d1 = list(map(lambda token: tags_to_id[tag_map[token]], delta1))
    ids_d2 = list(map(lambda token: tags_to_id[tag_map[token]], delta2))
    ids_st = list(map(lambda token: tags_to_id[tag_map[token]], start))   # start ids
    ids_en = list(map(lambda token: tags_to_id[tag_map[token]], end))     # end ids

    if ids_d1[0] == ids_d2[0]:
        test1 += 1
        # TODO resolve case in which both have same placeholder, use vector similarities, or none
        # TODO count it
    else:
        test2 += 1
        train_start.append(ids_st)
        train_end.append(ids_en)
        train_delta1.append(ids_d1)
        train_delta2.append(ids_d2)

# For the ARRANGE neural network
train_arrange_x = []
train_arrange_y = []

def prepare_arrange_tags(tags1, tags2):
    tags1 = tags1.split()
    tags2 = tags2.split()

    ids1 = list(map(lambda tag: tags_to_id[tag], tags1))
    ids2 = list(map(lambda tag: tags_to_id[tag], tags2))

    # TODO count identical ARRANGE tags signature
    train_arrange_x.append(ids1)
    train_arrange_y.append(1.)

    train_arrange_x.append(ids2)
    train_arrange_y.append(0.)

def train(p1, p2, error_type, t1, t2):
    # note: learn words is done in the error classifier (classification requires knowing the words)
    learn_word_frequencies(p1)  # only train frequencies on first part, second part is corrupted text
    if ENABLE_LEARN_WORDS:
        learn_words(p1)
    if error_type == 'REPLACE' and ENABLE_TRAIN_REPLACE_NN and not TRAIN_FROM_DISK_REPLACE:
        prepare_tags_for_nn(p1, p2, t1, t2)
    if ENABLE_TRAIN_ARRANGE_NN and ENABLE_PROCESS_ARRANGE_DATA and error_type == 'ARRANGE':
        prepare_arrange_tags(t1, t2)



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
    FILE_NAME = 'train'
    if not ONLY_TRAIN_NN:
        with open(FILE_NAME+'.txt', encoding='utf-8') as file, open(FILE_NAME + '.spacy.txt') as ftags:
            progress = 0
            start_time = time.time()
            words_processed = 0
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

                # Display progression in number of samples processed, use random to avoid too many (slow) interactions w/
                # console
                words_processed += len(p1.split()) + len(p2.split())
                if progress % 100 == 0:
                    print('\rProgress: [{}] Word Processed: [{}] Words per second: [{}] Lines per second: [{}]'
                          .format(progress, words_processed,
                                  words_processed / (time.time() - start_time), (progress / (time.time() - start_time)))
                          , end='')
        if ENABLE_LEARN_WORDS:
            save_learned_words()
        else:
            assert len(learned_words) == 0
        save_word_frequencies()
        print(test1, test2)
    if ENABLE_TRAIN_REPLACE_NN:
        # create the dataset

        max_start = 0
        max_end = 0
        samples = 0
        if not TRAIN_FROM_DISK_REPLACE:
            # saves the data to a file
            assert len(train_delta1) == len(train_delta2) == len(train_start) == len(train_end)
            max_start   = len(max(train_start, key=len))
            max_end     = len(max(train_end, key=len))
            samples = len(train_delta1)
            with open(FILE_NAME+'.replace.txt', 'x') as file_replace:
                file_replace.write('{} {} {}\n'.format(max_start, max_end, samples))
                for i in range(samples):
                    file_replace.write(' '.join(map(str, train_start[i]))  + '\t')
                    file_replace.write(' '.join(map(str, train_end[i]))    + '\t')
                    file_replace.write(str(train_delta1[i][0]) + '\t')
                    file_replace.write(str(train_delta2[i][0]) + '\n')


        def replace_nn_generator():
            with open(FILE_NAME+'.replace.txt') as file_replace:
                file_replace.readline()
                for replace_line in file_replace:
                    start, end, delta1, delta2 = replace_line.rstrip().split('\t')
                    start = list(map(int, start.split()))
                    end = list(map(int, end.split()))
                    delta1 = [int(delta1)]
                    delta2 = [int(delta2)]

                    [start] = keras.preprocessing.sequence.pad_sequences([start], maxlen=max_start)
                    [end]   = keras.preprocessing.sequence.pad_sequences([end]  , maxlen=max_end)

                    yield {'start': start, 'end': end, 'delta': delta1}, 1.
                    yield {'start': start, 'end': end, 'delta': delta2}, 0.


        with open(FILE_NAME+'.replace.txt') as file_replace:
            max_start, max_end, samples = list(map(int, file_replace.readline().strip().split()))
        dataset = tf.data.Dataset.from_generator(replace_nn_generator,
                 ({'start':tf.int32, 'end':tf.int32, 'delta': tf.int32}, tf.float32),
                 ({'start':tf.TensorShape([None,]), 'end':tf.TensorShape([None,]), 'delta': tf.TensorShape([1,])},
                    tf.TensorShape([])))


        dataset = dataset.repeat()
        dataset = dataset.shuffle(1000, seed=123)

        validation_dataset = dataset.take(int(samples * 0.1)) # 10% used for validation
        validation_dataset = validation_dataset.batch(1000)
        validation_dataset = validation_dataset.repeat()

        BATCH_SIZE = 1000
        dataset = dataset.batch(BATCH_SIZE)
        dataset.shuffle(10000)
        dataset.shuffle(10000)
        dataset.shuffle(10000)

        # Create the model
        input_start = tf.keras.layers.Input(shape=(max_start,), dtype=tf.int32, name='start')
        input_end   = tf.keras.layers.Input(shape=(max_end,), dtype=tf.int32, name='end')
        input_delta = tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name='delta')


        # input vocab is only 56 words
        embedding = tf.keras.layers.Embedding(output_dim = 20, input_dim=len(tags_to_id) + 1)

        x_s = embedding(input_start)
        x_e = embedding(input_end)
        x_d = embedding(input_delta)

        x_s = tf.keras.layers.LSTM(10)(x_s)
        x_e = tf.keras.layers.LSTM(10, go_backwards=False)(x_e) # converge such that last input is closest to the delta

        x_se = tf.keras.layers.concatenate([x_s, x_e])
        x_se = tf.keras.layers.Dense(10)(x_se)

        x_d = tf.keras.layers.Flatten()(x_d)

        x = tf.keras.layers.concatenate([x_se, x_d])
        x = tf.keras.layers.Dense(10)(x)
        output = tf.keras.layers.Dense(1)(x)

        model = tf.keras.Model(inputs=[input_start, input_end, input_delta], outputs=output)
        model.compile(optimizer=tf.train.AdamOptimizer(), loss='binary_crossentropy', metrics=['accuracy'])

        print(model.summary())
        print('-------------')
        print(dataset)
        print(dataset.output_shapes)
        print(dataset.output_types)

        checkpoint_path = 'checkpoints/%s_replace.ckpt' % FILE_NAME
        cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, save_best_only=False, verbose=1)

        model.load_weights(checkpoint_path)
        model.fit(dataset, steps_per_epoch=50, epochs=200, verbose=2, validation_data=validation_dataset, validation_steps=1, callbacks=[cp_callback])
    if ENABLE_TRAIN_ARRANGE_NN:
        # create the dataset
        samples = 0
        max_length = 0
        if ENABLE_PROCESS_ARRANGE_DATA:
            # saves the data to a file
            assert len(train_arrange_x) == len(train_arrange_y)
            max_length = len(max(train_arrange_x, key=len))
            samples = len(train_arrange_x)
            with open(FILE_NAME+'.arrange.txt', 'x') as file_arrange:
                file_arrange.write('{} {}\n'.format(max_length, samples))
                for i in range(samples):
                    file_arrange.write(' '.join(map(str, train_arrange_x[i])) + '\t')
                    file_arrange.write(str(train_arrange_y[i]) + '\n')


        def arrange_nn_generator():
            with open(FILE_NAME+'.arrange.txt') as file_arrange:
                file_arrange.readline()
                for arrange_line in file_arrange:
                    x, y = arrange_line.rstrip().split('\t')
                    x = list(map(int, x.split()))
                    y = int(y)

                    [x] = keras.preprocessing.sequence.pad_sequences([x], maxlen=max_length)
                    yield x, y


        with open(FILE_NAME+'.arrange.txt') as file_arrange:
            max_length, samples = list(map(int, file_arrange.readline().strip().split()))
        dataset = tf.data.Dataset.from_generator(arrange_nn_generator,
                 (tf.int32, tf.float32),
                 (tf.TensorShape([None,]), tf.TensorShape([])))

        dataset = dataset.repeat()
        dataset = dataset.shuffle(1000)

        validation_dataset = dataset.take(int(samples * 0.1))  # 10% used for validation
        validation_dataset = validation_dataset.batch(int(samples * 0.1))

        BATCH_SIZE = 10
        dataset = dataset.batch(BATCH_SIZE)


        input = tf.keras.layers.Input(shape=(None,), dtype=tf.int32)

        # input vocab is only 56 words
        x = tf.keras.layers.Embedding(output_dim = 20, input_dim=len(tags_to_id) + 1)(input)

        x = tf.keras.layers.LSTM(10, return_sequences=True)(x)
        x = tf.keras.layers.LSTM(10)(x)
        x = tf.keras.layers.Dense(10)(x)

        output = tf.keras.layers.Dense(1)(x)

        model = tf.keras.Model(inputs=input, outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        print(model.summary())
        print('-------------')
        print(dataset)
        print(dataset.output_shapes)
        print(dataset.output_types)



        model.fit(dataset, steps_per_epoch=100, epochs=20, verbose=2, validation_data=validation_dataset,
                  validation_steps=1)


