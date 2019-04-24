import tensorflow as tf
from tensorflow import keras

from Trainer import PATH_REPLACE_DATA, prepare_dataset

ENABLE_LOAD_CHECKPOINT = False
PATH_CHECKPOINT1 = 'nn1.ckpt'

# The search for the best REPLACE neural network
# Numbered by version

def main():
    pass

def create_network1():
    pass

def train_network1():
    def replace_nn_generator():
        with open(PATH_REPLACE_DATA) as file_replace:
            file_replace.readline()
            for replace_line in file_replace:
                start, end, delta1, delta2 = replace_line.rstrip().split('\t')
                start = list(map(int, start.split()))
                end = list(map(int, end.split()))
                delta1 = [int(delta1)]
                delta2 = [int(delta2)]

                [start] = keras.preprocessing.sequence.pad_sequences([start], maxlen=max_start)
                [end] = keras.preprocessing.sequence.pad_sequences([end], maxlen=max_end)

                yield {'start': start, 'end': end, 'delta': delta1}, 1.
                yield {'start': start, 'end': end, 'delta': delta2}, 0.

    with open(PATH_REPLACE_DATA) as file_replace:
        max_start, max_end, samples = list(map(int, file_replace.readline().strip().split()))
    dataset = tf.data.Dataset.from_generator(replace_nn_generator,
                                             ({'start': tf.int32, 'end': tf.int32, 'delta': tf.int32}, tf.float32),
                                             ({'start': tf.TensorShape([None, ]), 'end': tf.TensorShape([None, ]),
                                               'delta': tf.TensorShape([1, ])},
                                              tf.TensorShape([])))

    dataset, validation_dataset = prepare_dataset(dataset, samples, batch_size=1024)

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

main()