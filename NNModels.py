import tensorflow as tf
from tensorflow import keras

from NeuralNetworkHelper import tags_to_id


def create_replace_nn_model():
    input_start = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='start')
    input_end = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='end')
    input_delta = tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name='delta')

    # input vocab is only 56 words
    embedding = tf.keras.layers.Embedding(output_dim=30, input_dim=len(tags_to_id) + 1)

    x_s = embedding(input_start)
    x_e = embedding(input_end)
    x_d = embedding(input_delta)

    x_s = tf.keras.layers.CuDNNLSTM(20, return_sequences=True)(x_s)
    x_s = tf.keras.layers.CuDNNLSTM(20)(x_s)
    x_e = tf.keras.layers.CuDNNLSTM(20, return_sequences=True, go_backwards=False)(x_e)
    x_e = tf.keras.layers.CuDNNLSTM(20, go_backwards=False)(x_e)

    x_se = tf.keras.layers.concatenate([x_s, x_e])
    x_se = tf.keras.layers.Dense(20, activation=tf.nn.relu)(x_se)
    x_se = tf.keras.layers.Dense(20, activation=tf.nn.tanh)(x_se)

    x_d = tf.keras.layers.Flatten()(x_d)

    x = tf.keras.layers.concatenate([x_se, x_d])
    x = tf.keras.layers.Dense(20, activation=tf.nn.relu)(x)
    x = tf.keras.layers.Dense(20, activation=tf.nn.tanh)(x)
    output = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(x)

    model = tf.keras.Model(inputs=[input_start, input_end, input_delta], outputs=output)
    return model


def create_arrange_nn_model():
    input = tf.keras.layers.Input(shape=(None,), dtype=tf.int32)

    # input vocab is only 56 words
    x = tf.keras.layers.Embedding(output_dim=30, input_dim=len(tags_to_id) + 1)(input)

    x = tf.keras.layers.CuDNNLSTM(20, return_sequences=True)(x)
    x = tf.keras.layers.CuDNNLSTM(20)(x)
    x = tf.keras.layers.Dense(20, activation=tf.nn.tanh)(x)

    output = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(x)

    model = tf.keras.Model(inputs=input, outputs=output)
    return model


def create_replace1_nn_model():
    BATCH_SIZE = int(256 / 2) # TODO acquire this variable automatically
    input_start = keras.layers.Input(shape=(None, 300), dtype=tf.float32, name='start')
    input_end = keras.layers.Input(shape=(None, 300), dtype=tf.float32, name='end')
    input_delta1 = keras.layers.Input(shape=(300,), dtype=tf.float32, name='delta1')
    input_delta2 = keras.layers.Input(shape=(300,), dtype=tf.float32, name='delta2')

    # Cannot do masking while unknown word vectors default to 0.
    # masking = keras.layers.Masking(0.)
    # input_start = masking(input_start)
    # input_end   = masking(input_end)
    dense_reduce_dim = keras.layers.Dense(50, activation=tf.nn.tanh)

    x_s = keras.layers.TimeDistributed(dense_reduce_dim)(input_start)
    x_e = keras.layers.TimeDistributed(dense_reduce_dim)(input_end)
    x_d1 = dense_reduce_dim(input_delta1)
    x_d2 = dense_reduce_dim(input_delta2)

    x_s = keras.layers.CuDNNLSTM(20, return_sequences=True)(x_s)
    x_s = keras.layers.Dropout(0.1, noise_shape=(BATCH_SIZE * 2, 1, 20))(x_s)
    x_s = keras.layers.CuDNNLSTM(20)(x_s)

    x_e = keras.layers.CuDNNLSTM(20, return_sequences=True, go_backwards=True)(x_e)
    x_e = keras.layers.Dropout(0.1, noise_shape=(BATCH_SIZE * 2, 1, 20))(x_e)
    x_e = keras.layers.CuDNNLSTM(20)(x_e)

    x_se = keras.layers.concatenate([x_s, x_e])
    x_se = keras.layers.Dense(20, kernel_regularizer=keras.regularizers.l2(0.01))(x_se)

    dense_del = keras.layers.Dense(20)
    x_d1 = dense_del(x_d1)
    x_d2 = dense_del(x_d2)

    x = keras.layers.concatenate([x_se, x_d1, x_d2])
    x = keras.layers.Dense(30, activation=tf.nn.tanh)(x)
    # x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Dense(20, activation=tf.nn.tanh)(x)
    # x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Dense(1, activation=tf.nn.sigmoid)(x)

    output = x

    model = tf.keras.Model(inputs=[input_start, input_end, input_delta1, input_delta2], outputs=output)
    return model


def create_replace2_nn_model():
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
    x = keras.layers.concatenate([x1, x2])

    # LSTMs with merged inputs
    x = keras.layers.CuDNNLSTM(20)(x)

    # Final dense layers
    x = keras.layers.Dense(20, activation=tf.nn.tanh)(x)
    x = keras.layers.Dense(1, activation=tf.nn.sigmoid)(x)

    output = x

    model = tf.keras.Model(inputs=[input_part1, input_part2], outputs=output)
    return model


def create_nn_model(name):
    return globals()['create_%s_nn_model' % name]()