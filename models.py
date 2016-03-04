from __future__ import absolute_import
from __future__ import print_function

import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedMerge
from keras.layers.embeddings import Embedding
from keras.datasets import imdb
from keras.constraints import unitnorm
from sentiment_classification import nn_input
from keras.layers.core import Reshape, Flatten, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D
from keras.utils.visualize_util import plot
from keras.callbacks import EarlyStopping
from keras.regularizers import l2

'''
    Train a deep averaging network (DAN) using keras.
    The model is described in "Deep Unordered Composition Rivals Syntactic Methods
    for Text Classification" by Iyyer, Manjunatha, Boyd-Graber, and Daume (ACL 2015).
    Notes:
    - Deep averaging networks can be thought of as a drop-in for recurrent neural networks
    - In the paper, dropout was applied at the word level; that's not done here. I just added dropout
    in the fully connected layers. This could be fixed later.
    - The sequence of characters doesn't matter, so it's pretty surprising that something like
    this does well.
    - Parameters are not optimized in any way -- I just used a fixed number for the embedding
    and hidden dimension
    This model achieves 0.8340 test accuracy after 3 epochs for IMDB sentiment classification.
    Input Data format:
        X: a list of number indicating the word index, e.g. [12 54 63 74 12 10]
        Y: a list of 0 or 1 indicating the labels of sentence, e.g. [0 1 0 1 0 1 0 0 1]
'''


def dan(weights=None):
    '''
    DAN model with pre-trained embeddings
    :param max_features: the number of words
    :return: keras model
    '''
    max_features = weights.shape[0]  # weights.shape = (vocabulary size, vector dimension)
    print('Build model...')
    model = Sequential()
    model.add(Embedding(input_dim=max_features, output_dim=300, weights=[weights], W_regularizer=l2(1e-5)))
    model.add(Dropout(.5))
    model.add(TimeDistributedMerge(mode='ave'))
    model.add(Dense(input_dim=300, output_dim=300, activation='relu', W_regularizer=l2(1e-5), b_regularizer=l2(1e-5)))
    model.add(Dropout(.4))
    model.add(Dense(input_dim=300, output_dim=300, activation='relu', W_regularizer=l2(1e-5), b_regularizer=l2(1e-5)))
    model.add(Dropout(.2))
    # model.add(Dense(input_dim=300, output_dim=300, activation = 'relu', W_regularizer=l2(1e-5), b_regularizer=l2(1e-5)))
    # model.add(Dropout(.2))
    model.add(Dense(input_dim=300, output_dim=2, activation='softmax', W_regularizer=l2(1e-5), b_regularizer=l2(1e-5)))
    return model


def cnn(W=None):
    # Number of feature maps (outputs of convolutional layer)
    N_fm = 20
    dense_nb = 20
    # kernel size of convolutional layer
    kernel_size = 5
    conv_input_width = W.shape[1]   # dims=300
    conv_input_height = 200  # maxlen of sentence

    model = Sequential()
    # Embedding layer (lookup table of trainable word vectors)
    model.add(Embedding(input_dim=W.shape[0], output_dim=W.shape[1], weights=[W], W_constraint=unitnorm()))
    # Reshape word vectors from Embedding to tensor format suitable for Convolutional layer
    model.add(Reshape(dims=(1, conv_input_height, conv_input_width)))

    # first convolutional layer
    model.add(Convolution2D(nb_filter=N_fm, nb_row=kernel_size, nb_col=conv_input_width, border_mode='valid',
                            W_regularizer=l2(0.0001), activation='relu'))
    # ReLU activation
    model.add(Dropout(0.5))

    # aggregate data in every feature map to scalar using MAX operation
    # model.add(MaxPooling2D(pool_size=(conv_input_height-kernel_size+1, 1), border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(kernel_size * 5, 1), border_mode='valid'))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(output_dim=dense_nb, activation='relu'))
    model.add(Dropout(0.2))
    # Inner Product layer (as in regular neural network, but without non-linear activation function)
    model.add(Dense(output_dim=2, activation='softmax'))
    # SoftMax activation; actually, Dense+SoftMax works as Multinomial Logistic Regression
    return model


def imdb_cnn(W=None):
    # Number of feature maps (outputs of convolutional layer)
    N_fm = 60
    # kernel size of convolutional layer
    kernel_size = 3
    dims = 300  # 300 dimension
    maxlen = 200  # maxlen of sentence
    max_features = W.shape[0]
    hidden_dims = 100
    print('Build model...')
    model = Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(max_features, dims, input_length=maxlen, weights=[W]))
    model.add(Dropout(0.5))

    # we add a Convolution1D, which will learn nb_filter
    # word group filters of size filter_length:
    model.add(Convolution1D(nb_filter=N_fm,
                            filter_length=kernel_size,
                            border_mode='valid',
                            activation='relu',
                            ))
    model.add(Dropout(0.5))
    # we use standard max pooling (halving the output of the previous layer):
    model.add(MaxPooling1D(pool_length=kernel_size*7))

    # We flatten the output of the conv layer,
    # so that we can add a vanilla dense layer:
    model.add(Flatten())

    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model

def imdb_cnn_sts(W=None):
    # Number of feature maps (outputs of convolutional layer)
    N_fm = 20
    # kernel size of convolutional layer
    kernel_size = 3
    dims = 300  # 300 dimension
    maxlen = 200  # maxlen of sentence
    max_features = W.shape[0]
    hidden_dims = 20
    print('Build model...')
    model = Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(max_features, dims, input_length=maxlen, weights=[W]))
    model.add(Dropout(0.5))

    # we add a Convolution1D, which will learn nb_filter
    # word group filters of size filter_length:
    model.add(Convolution1D(nb_filter=N_fm,
                            filter_length=kernel_size,
                            border_mode='valid',
                            activation='relu',
                            ))
    model.add(Dropout(0.5))
    # we use standard max pooling (halving the output of the previous layer):
    model.add(MaxPooling1D(pool_length=kernel_size))

    # We flatten the output of the conv layer,
    # so that we can add a vanilla dense layer:
    model.add(Flatten())

    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(2))
    model.add(Activation('softmax'))
    return model


def Deep_CNN(W):
    # Number of feature maps (outputs of convolutional layer)
    N_fm = 32
    # kernel size of convolutional layer
    kernel_size = 3
    conv_input_width = W.shape[1]  # 300 dimension
    conv_input_height = 200  # maxlen of sentence

    # Two Convolutional Layers with Pooling Layer
    model = Sequential()
    model.add(Embedding(input_dim=W.shape[0], output_dim=W.shape[1], weights=[W]))
    # Reshape word vectors from Embedding to tensor format suitable for Convolutional layer
    model.add(Reshape(dims=(1, conv_input_height, conv_input_width)))
    model.add(Convolution2D(nb_filter=N_fm, nb_row=kernel_size, nb_col=conv_input_width, border_mode='valid',
                            activation='relu'))
    model.add(Dropout(0.5))
    # model.add(Convolution2D(nb_filter=N_fm, nb_row=kernel_size, nb_col=1, border_mode='valid', activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Convolution2D(nb_filter=N_fm, nb_row=kernel_size, nb_col=1, border_mode='valid', activation='relu'))
    # model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Flatten())
    model.add(Dense(output_dim=N_fm, activation='relu'))
    model.add(Dropout(0.5))
    # Fully Connected Layer as output layer
    model.add(Dense(output_dim=2, activation='softmax'))
    return model


def hybrid_model(W):
    '''
    This function return a hybrid model of cnn and dan
    :param W: initial weights of the embedding layer
    :return: model
    '''
    max_features = W.shape[0]
    N_fm = 300
    # kernel size of convolutional layer
    kernel_size = 8
    conv_input_width = W.shape[1]
    conv_input_height = 200  # maxlen of sentence

    cnn = Sequential()
    cnn.add(Embedding(input_dim=max_features, output_dim=300, weights=[W]))
    cnn.add(Dropout(.5))
    cnn.add(Reshape(dims=(1, conv_input_height, conv_input_width)))
    # first convolutional layer
    cnn.add(Convolution2D(nb_filter=N_fm, nb_row=kernel_size, nb_col=conv_input_width, border_mode='valid',
                          W_regularizer=l2(0.0001), activation='relu'))
    # ReLU activation
    cnn.add(Dropout(0.5))
    # aggregate data in every feature map to scalar using MAX operation
    cnn.add(MaxPooling2D(pool_size=(conv_input_height - kernel_size + 1, 1), border_mode='valid'))
    cnn.add(Dropout(0.5))
    cnn.add(Flatten())
    cnn.add(Dense(output_dim=N_fm, activation='relu'))

    dan = Sequential()
    dan.add(Embedding(input_dim=max_features, output_dim=300, weights=[W]))
    dan.add(Dropout(.5))
    dan.add(TimeDistributedMerge(mode='ave'))
    dan.add(Dense(input_dim=300, output_dim=300, activation='relu'))
    dan.add(Dropout(.5))
    dan.add(Dense(input_dim=300, output_dim=300, activation='relu'))
    dan.add(Dropout(.5))
    dan.add(Dense(input_dim=300, output_dim=300, activation='relu'))

    model = Sequential()
    model.add(Merge([cnn, dan], mode='sum'))
    model.add(Dense(300, activation='relu'))
    dan.add(Dropout(.5))
    model.add(Dense(2, activation='softmax'))
    return model

    # The input data of this function:
    # model.fit([X_train,X_train], y_train, batch_size=batch_size, nb_epoch=100, validation_data=([X_test,X_test], y_test), show_accuracy=True, callbacks=[early_stopping])
    # score, acc = model.evaluate([X_test,X_test], y_test, batch_size=batch_size, show_accuracy=True)


# def grid_search()


# https://github.com/fchollet/keras/issues/233
def ngrams_cnn(W):
    pass


def parallel_cnn(W):
    (nb_vocab, dims) =W.shape
    N_filter=20

    filter_shapes = [[2, 300], [3, 300], [4, 300], [5, 300]]
    pool_shapes = [[25, 1], [24, 1], [23, 1], [22, 1]]  # Four Parallel Convolutional Layers with Four Pooling Layers
    model = Sequential()
    sub_models = []
    for i in range(len(pool_shapes)):
        pool_shape = pool_shapes[i]
        filter_shape = filter_shapes[i]
        sub_model = Sequential()
        sub_model.add(Embedding(input_dim=nb_vocab, output_dim=dims, weights=[W], W_constraint=unitnorm()))
        # Reshape word vectors from Embedding to tensor format suitable for Convolutional layer
        sub_model.add(Reshape(dims=(1, 200, dims)))
        sub_model.add(Convolution2D(nb_filter=N_filter, nb_row=filter_shape[0], nb_col=filter_shape[1], border_mode = 'valid', activation = 'relu'))
        sub_model.add(MaxPooling2D(pool_size=(pool_shape[0], pool_shape[1]), border_mode='valid'))
        sub_model.add(Flatten())
        sub_models.append(sub_model)

    model.add((Merge(sub_models, mode='concat')))
    # Fully Connected Layer with dropout
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    # Fully Connected Layer as output layer
    model.add(Dense(2, activation='softmax'))
    return model


def SA_sst():
    ((x_train_idx_data, y_train_valence, y_train_labels,
      x_test_idx_data, y_test_valence, y_test_labels,
      x_valid_idx_data, y_valid_valence, y_valid_labels,
      x_train_polarity_idx_data, y_train_polarity,
      x_test_polarity_idx_data, y_test_polarity,
      x_valid_polarity_idx_data, y_valid_polarity), W) = nn_input()                    #  build_keras_input_amended or build_keras_input

    maxlen = 200  # cut texts after this number of words (among top max_features most common words)
    batch_size = 16
    (X_train, y_train), (X_test, y_test), (X_valid, y_valide) = (x_train_polarity_idx_data, y_train_polarity), (
    x_test_polarity_idx_data, y_test_polarity), (x_valid_polarity_idx_data, y_valid_polarity)
    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')
    # m= 0
    # for i in X_train:
    #     if len(i) >0:
    #         for j in i:
    #             if j > m:
    #                 m=j
    # print(m)
    max_features = W.shape[0]  # shape of W: (13631, 300) , changed to 14027 through min_df = 3

    print("Pad sequences (samples x time)")
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    X_valid = sequence.pad_sequences(X_valid, maxlen=maxlen)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    nb_classes = 2
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    y_valide = np_utils.to_categorical(y_valide, nb_classes)

    model = Deep_CNN(W)
    plot(model, to_file='./images/model.png')

    # try using different optimizers and different optimizer configs
    # model.compile(loss='binary_crossentropy', optimizer='adagrad', class_mode="binary")
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='adagrad')   # adagrad

    print("Train...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=20, validation_data=(X_test, y_test), show_accuracy=True,
              callbacks=[early_stopping])
    score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, show_accuracy=True)
    print('Test score:', score)
    print('Test accuracy:', acc)


def imdb_test():
    # set parameters:
    max_features = 5000  # number of vocabulary
    maxlen = 200  # padding
    batch_size = 16
    nb_epoch = 10

    print('Loading data...')
    (X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features,
                                                          test_split=0.2)

    print("Pad sequences (samples x time)")
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    nb_classes = 2
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    model = imdb_cnn()
    plot(model, to_file='./images/imdb_model.png')

    # try using different optimizers and different optimizer configs
    # model.compile(loss='binary_crossentropy', optimizer='adagrad', class_mode="binary")
    model.compile(loss='categorical_crossentropy', optimizer='adagrad')

    print("Train...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_test, y_test),
              show_accuracy=True, callbacks=[early_stopping])
    score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, show_accuracy=True)
    print('Test score:', score)
    print('Test accuracy:', acc)


if __name__ == "__main__":
    # imdb_test()
    # exit()
    SA_sst()
