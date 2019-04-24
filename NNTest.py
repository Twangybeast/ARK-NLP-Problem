import tensorflow as tf
from tensorflow import keras

from Trainer import FILE_NAME, PATH_REPLACE_DATA, prepare_dataset
from ErrorClassifier import tokenize, find_all_delta_from_tokens
from NNModels import create_nn_model

ENABLE_LOAD_CHECKPOINT = False
ENABLE_SAVE_TFRECORD = False

PATH_CHECKPOINT1 = 'nn1.ckpt'
PATH_CHECKPOINT2 = 'nn2.ckpt'

PATH_REPLACE_RAW = FILE_NAME+'.replace.original.txt'
PATH_TFRECORD_REPLACE = FILE_NAME + '1.replace.tfrecord'

# The search for the best REPLACE neural network
# Numbered by version


def main():
    train_network1()


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

        # doubles the batch size, with the second half switching the deltas and reversing the label
        start = tf.concat([start, start], axis=0)
        end   = tf.concat([end, end], axis=0)
        delta1 = tf.concat([d1, d2], axis=0)
        delta2 = tf.concat([d2, d1], axis=0)

        y1 = tf.fill(tf.gather(tf.shape(x), 0), 0.)
        y2 = tf.fill(tf.gather(tf.shape(x), 0), 1.)
        y = tf.concat(y1, y2, axis=0)
        return {'start': start, 'end': end, 'delta1':delta1, 'delta2':delta2}, y

    dataset = tf.data.TFRecordDataset(PATH_TFRECORD_REPLACE)
    dataset = dataset.map(decode, num_parallel_calls=8)
    dataset = dataset.shuffle(100000, seed=123)
    dataset = dataset.padded_batch(1024/2, {'start': (None, 300), 'end': (None, None), 'delta1': (None,), 'delta2': (None,)})
    dataset = dataset.prefetch(1024/2)
    dataset = dataset.map(add_label, num_parallel_calls=8)

    # dataset.shuffle(1000, seed=123)
    validation_dataset = dataset.take(10)

    # Create the model
    model = create_nn_model('replace1')
    model.compile(optimizer=tf.train.AdamOptimizer(), loss='binary_crossentropy', metrics=['accuracy'])

    print(model.summary())
    print('-------------')

    cp_callback = tf.keras.callbacks.ModelCheckpoint(PATH_CHECKPOINT1, save_weights_only=True,
                                                     save_best_only=False, verbose=1)
    if ENABLE_LOAD_CHECKPOINT:
        model.load_weights(PATH_CHECKPOINT1)

    model.fit(dataset, steps_per_epoch=50, epochs=200, verbose=2, validation_data=validation_dataset,
              validation_steps=1, callbacks=[cp_callback])


def create_network2():
    input_part1 = keras.layers.Input(shape=(None,300), dtype=tf.float32, name='part1')
    input_part2 = keras.layers.Input(shape=(None,300), dtype=tf.float32, name='part2')

    # Cannot do masking while unknown word vectors default to 0.
    # masking = keras.layers.Masking(0.)
    # input_start = masking(input_start)
    # input_end   = masking(input_end)

    # reduce dimensions of word vectors
    dense_reduce_dim = keras.layers.TimeDistributed(keras.layers.Dense(50, activation=tf.nn.tanh))
    x1 = dense_reduce_dim(input_part1)
    x2 = dense_reduce_dim(input_part2)

    # LSTM, recurrent natures
    lstm1 = keras.layers.CuDNNLSTM(20, return_sequences=True)
    x1 = lstm1(x1)
    x2 = lstm1(x2)

    # merge layer
    x = keras.layers.concatenate([x1, x2], axis=-2)

    # LSTMs with merged inputs
    x = keras.layers.CuDNNLSTM(20)(x)

    # Final dense layers
    x = keras.layers.Dense(20, activation=tf.nn.tanh)(x)
    x = keras.layers.Dense(1, activation=tf.nn.sigmoid)(x)

    output = x

    model = tf.keras.Model(inputs=[input_part1, input_part2], outputs=output)
    return model


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
        start = tf.sparse.to_dense(sequence['s'])
        end   = tf.sparse.to_dense(sequence['e'])
        delta1 = tf.expand_dims(sequence['d1'], axis=0)
        delta2 = tf.expand_dims(sequence['d2'], axis=0)

        part1 = tf.concat([start, delta1, end], axis=0)
        part2 = tf.concat([start, delta2, end], axis=0)
        return {'part1': part1, 'part2': part2}
    def add_label(x):
        p1 = x['part1']
        p2 = x['part2']

        # doubles the batch size, with the second half switching the parts and reversing the label
        part1 = tf.concat([p1, p2], axis=0)
        part2 = tf.concat([p2, p1], axis=0)

        y1 = tf.fill(tf.gather(tf.shape(x), 0), 0.)
        y2 = tf.fill(tf.gather(tf.shape(x), 0), 1.)
        y = tf.concat(y1, y2, axis=0)
        return {'part1': part1, 'part2': part2}, y

    dataset = tf.data.TFRecordDataset(PATH_TFRECORD_REPLACE)
    dataset = dataset.map(decode, num_parallel_calls=8)
    dataset = dataset.shuffle(100000, seed=123)
    dataset = dataset.padded_batch(1024/2, {'start': (None, 300), 'end': (None, None), 'delta1': (None,), 'delta2': (None,)})
    dataset = dataset.prefetch(1024/2)
    dataset = dataset.map(add_label, num_parallel_calls=8)

    # dataset.shuffle(1000, seed=123)
    validation_dataset = dataset.take(10)

    # Create the model
    model = create_network2()
    model.compile(optimizer=tf.train.AdamOptimizer(), loss='binary_crossentropy', metrics=['accuracy'])

    print(model.summary())
    print('-------------')

    cp_callback = tf.keras.callbacks.ModelCheckpoint(PATH_CHECKPOINT2, save_weights_only=True,
                                                     save_best_only=False, verbose=1)
    if ENABLE_LOAD_CHECKPOINT:
        model.load_weights(PATH_CHECKPOINT2)

    model.fit(dataset, steps_per_epoch=50, epochs=200, verbose=2, validation_data=validation_dataset,
              validation_steps=1, callbacks=[cp_callback])


if __name__ == '__main__':
    main()

