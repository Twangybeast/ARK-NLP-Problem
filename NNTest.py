import tensorflow as tf
from tensorflow import keras

from NeuralNetworkHelper import FILE_NAME
from NNModels import create_nn_model

ENABLE_LOAD_CHECKPOINT = False
ENABLE_SAVE_TFRECORD = False

if ENABLE_SAVE_TFRECORD:
    from TokenHelper import tokenize, find_all_delta_from_tokens

PATH_CHECKPOINT1 = 'checkpoints/nn1_1.ckpt'
PATH_CHECKPOINT2 = 'checkpoints/nn2.ckpt'

PATH_REPLACE_RAW = FILE_NAME+'.replace.original.txt'
PATH_TFRECORD_REPLACE = FILE_NAME + '1.replace.tfrecord'



# The search for the best REPLACE neural network
# Numbered by version


def main():
    train_network1()
    # train_network2()


def train_network1():
    if ENABLE_SAVE_TFRECORD:
        # Create the TFRecord
        def token_to_feature(token):
            # Might give errors if no "if" statement
            vector = token.vector # if token.has_vector else [0.] * 300 # Assume default word vector length of 300
            return tf.train.Feature(float_list=tf.train.FloatList(value=vector))
        # takes a partial sequence of token (e.g. start, delta1) and returns a FeatureList
        def tokens_to_list(tokens):
            features = []
            for token in tokens:
                features.append(token_to_feature(token))
            return tf.train.FeatureList(feature=features)
        def serialize_example(start, end, delta1, delta2):
            dict_feature = {
                's': tokens_to_list(start),
                'e': tokens_to_list(end)
            }
            dict_context = {
                'd1': token_to_feature(delta1[0]),
                'd2': token_to_feature(delta2[0])
            }
            return tf.train.SequenceExample(context=tf.train.Features(feature=dict_context),
                                            feature_lists=tf.train.FeatureLists(feature_list=dict_feature)).SerializeToString()
        with open(PATH_REPLACE_RAW, encoding='utf-8') as file, \
                tf.python_io.TFRecordWriter(PATH_TFRECORD_REPLACE) as writer:
            samples = int(file.readline().rstrip())
            for line in file:
                start, end, delta1, delta2 = line.rstrip('\n').split('\t')
                part1 = ' '.join([start, delta1, end])
                part2 = ' '.join([start, delta2, end])

                tokens1, tokens2 = tokenize(part1, part2)

                delta1, delta2, start, end = find_all_delta_from_tokens(tokens1, tokens2)

                example = serialize_example(start, end, delta1, delta2)
                writer.write(example)


    def decode(serialized):
        sequence_features = {
            's': tf.io.VarLenFeature(tf.float32),
            'e': tf.io.VarLenFeature(tf.float32)
        }
        context_features = {
            'd1': tf.io.FixedLenFeature([300], tf.float32),
            'd2': tf.io.FixedLenFeature([300], tf.float32)
        }
        context, sequence = tf.io.parse_single_sequence_example(serialized=serialized,
                                                sequence_features=sequence_features, context_features=context_features)
        start = tf.sparse.to_dense(sequence['s'])
        end   = tf.sparse.to_dense(sequence['e'])
        return {'start': start, 'end': end,'delta1':context['d1'], 'delta2':context['d2']}
    def add_label(x):
        start = x['start']
        end   = x['end']
        d1 = x['delta1']
        d2 = x['delta2']

        batch_size = tf.gather(tf.shape(start), 0)
        batch_size = tf.expand_dims(batch_size, axis=0)

        # doubles the batch size, with the second half switching the deltas and reversing the label
        start = tf.concat([start, start], axis=0)
        end   = tf.concat([end, end], axis=0)
        delta1 = tf.concat([d1, d2], axis=0)
        delta2 = tf.concat([d2, d1], axis=0)

        y1 = tf.fill(batch_size, 0.)
        y2 = tf.fill(batch_size, 1.)
        y = tf.concat([y1, y2], axis=0)
        return {'start': start, 'end': end, 'delta1':delta1, 'delta2':delta2}, y

    def decode_dataset(dataset):
        dataset = dataset.map(decode, num_parallel_calls=8)
        dataset = dataset.repeat()
        dataset = dataset.padded_batch(BATCH_SIZE,
                                       {'start': (None, None), 'end': (None, None), 'delta1': (None,),
                                        'delta2': (None,)})
        dataset = dataset.prefetch(int(BATCH_SIZE))
        dataset = dataset.map(add_label, num_parallel_calls=8)
        return dataset


    dataset = tf.data.TFRecordDataset(PATH_TFRECORD_REPLACE)
    BATCH_SIZE = int(256/2) # divide by 2 since it doubles while adding labels
    validation_dataset = dataset.take(BATCH_SIZE * 100)
    training_dataset = dataset.skip(BATCH_SIZE * 100)

    validation_dataset = validation_dataset.shuffle(BATCH_SIZE * 100, seed=123)
    training_dataset = training_dataset.shuffle(10000, seed=123)

    validation_dataset = decode_dataset(validation_dataset)
    training_dataset = decode_dataset(training_dataset)

    # Create the model
    model = create_nn_model('replace1')
    # run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)

    model.compile(optimizer=tf.train.AdamOptimizer(), loss='binary_crossentropy', metrics=['accuracy'])

    print(model.summary())
    print('-------------')

    cp_callback = tf.keras.callbacks.ModelCheckpoint(PATH_CHECKPOINT1, save_weights_only=True,
                                                     save_best_only=False, verbose=1)
    if ENABLE_LOAD_CHECKPOINT:
        model.load_weights(PATH_CHECKPOINT1)

    model.fit(training_dataset, steps_per_epoch=50 * 4, epochs=200, verbose=2, validation_data=validation_dataset,
              validation_steps=10, callbacks=[cp_callback])


def train_network2():
    def decode(serialized):
        sequence_features = {
            's': tf.io.VarLenFeature(tf.float32),
            'e': tf.io.VarLenFeature(tf.float32)
        }
        context_features = {
            'd1': tf.io.FixedLenFeature([300], tf.float32),
            'd2': tf.io.FixedLenFeature([300], tf.float32)
        }
        context, sequence = tf.io.parse_single_sequence_example(serialized=serialized,
                                                sequence_features=sequence_features, context_features=context_features)
        start = tf.reshape(tf.sparse.to_dense(sequence['s']), [-1, 300])
        end   = tf.reshape(tf.sparse.to_dense(sequence['e']), [-1, 300])
        delta1 = tf.expand_dims(context['d1'], axis=0)
        delta2 = tf.expand_dims(context['d2'], axis=0)

        part1 = tf.concat([start, delta1, end], axis=0)
        part2 = tf.concat([start, delta2, end], axis=0)
        return {'part1': part1, 'part2': part2}
    def add_label(x):
        p1 = x['part1']
        p2 = x['part2']

        batch_size = tf.gather(tf.shape(p1), 0)
        batch_size = tf.expand_dims(batch_size, axis=0)

        # doubles the batch size, with the second half switching the parts and reversing the label
        part1 = tf.concat([p1, p2], axis=0)
        part2 = tf.concat([p2, p1], axis=0)

        y1 = tf.fill(batch_size, 0.)
        y2 = tf.fill(batch_size, 1.)
        y = tf.concat([y1, y2], axis=0)
        return {'part1': part1, 'part2': part2}, y

    def decode_dataset(dataset):
        dataset = dataset.map(decode, num_parallel_calls=8)
        dataset = dataset.repeat()
        dataset = dataset.padded_batch(BATCH_SIZE, {'part1': (None, 300), 'part2': (None, 300)})
        dataset = dataset.prefetch(BATCH_SIZE)
        dataset = dataset.map(add_label, num_parallel_calls=8)
        return dataset
    dataset = tf.data.TFRecordDataset(PATH_TFRECORD_REPLACE)
    BATCH_SIZE = int(256/2) # divide by 2 since it doubles while adding labels

    validation_dataset = dataset.take(BATCH_SIZE * 100)
    training_dataset = dataset.skip(BATCH_SIZE * 100)

    validation_dataset = validation_dataset.shuffle(BATCH_SIZE * 100, seed=123)
    training_dataset = training_dataset.shuffle(10000, seed=123)

    validation_dataset = decode_dataset(validation_dataset)
    training_dataset = decode_dataset(training_dataset)

    # Create the model
    model = create_nn_model('replace2')
    model.compile(optimizer=tf.train.AdamOptimizer(), loss='binary_crossentropy', metrics=['accuracy'])

    print(model.summary())
    print('-------------')

    cp_callback = tf.keras.callbacks.ModelCheckpoint(PATH_CHECKPOINT2, save_weights_only=True,
                                                     save_best_only=False, verbose=1)
    if ENABLE_LOAD_CHECKPOINT:
        model.load_weights(PATH_CHECKPOINT2)

    model.fit(training_dataset, steps_per_epoch=50 * 4, epochs=200, verbose=2, validation_data=validation_dataset,
              validation_steps=1, callbacks=[cp_callback])


if __name__ == '__main__':
    main()

