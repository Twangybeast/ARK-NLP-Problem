import tensorflow as tf
from tensorflow import keras

from Trainer import FILE_NAME, PATH_REPLACE_DATA, prepare_dataset
from ErrorClassifier import tokenize, find_all_delta_from_tokens

ENABLE_LOAD_CHECKPOINT = False
ENABLE_SAVE_TFRECORD = False

PATH_CHECKPOINT1 = 'nn1.ckpt'

PATH_REPLACE_RAW = FILE_NAME+'.replace.original.txt'
PATH_TFRECORD_REPLACE1 = FILE_NAME + '1.replace.tfrecord'

# The search for the best REPLACE neural network
# Numbered by version

def main():
    train_network1()

def create_network1():
    pass

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
                tf.python_io.TFRecordWriter(PATH_TFRECORD_REPLACE1) as writer:
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
    dataset = tf.data.TFRecordDataset(PATH_TFRECORD_REPLACE1)
    dataset = dataset.map(decode, num_parallel_calls=8)
    dataset = dataset.shuffle(100000, seed=123)
    dataset = dataset.padded_batch(1024, {'start': (None, 300), 'end': (None, None), 'delta1': (None,), 'delta2': (None,)})
    dataset = dataset.prefetch(1024)

    # dataset.shuffle(1000, seed=123)
    validation_dataset = dataset.take(10)

    # Create the model
    model = create_network1()
    model.compile(optimizer=tf.train.AdamOptimizer(), loss='binary_crossentropy', metrics=['accuracy'])

    print(model.summary())
    print('-------------')

    cp_callback = tf.keras.callbacks.ModelCheckpoint(PATH_CHECKPOINT1, save_weights_only=True,
                                                     save_best_only=False, verbose=1)
    if ENABLE_LOAD_CHECKPOINT:
        model.load_weights(PATH_CHECKPOINT1)

    model.fit(dataset, steps_per_epoch=50, epochs=200, verbose=2, validation_data=validation_dataset,
              validation_steps=1, callbacks=[cp_callback])


# Copied below functions from the TFRecord Tensorflow tutorial
# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


if __name__ == '__main__':
    main()

