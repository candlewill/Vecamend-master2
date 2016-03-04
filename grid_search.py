from __future__ import absolute_import
from __future__ import print_function

from sklearn.grid_search import ParameterGrid
import numpy as np
from save_data import dump_picle
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedMerge
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.datasets import imdb
from keras.constraints import unitnorm
from sentiment_classification import build_keras_input
from keras.layers.core import Reshape, Flatten, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.regularizers import l2
from keras.utils.visualize_util import plot
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from load_data import load_pickle
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


def dan_pre_trained(weights=None, p1=0.5,p2=0.4,p3=0.2):
    '''
    DAN model with pre-trained embeddings
    :param max_features: the number of words
    :return: keras model
    '''
    max_features = weights.shape[0]      # weights.shape = (vocabulary size, vector dimension)
    print('Build model...')
    model = Sequential()
    model.add(Embedding(input_dim = max_features, output_dim = 300, weights=[weights], W_regularizer=l2(1e-5)))
    model.add(Dropout(p1))
    model.add(TimeDistributedMerge(mode='ave'))
    model.add(Dense(input_dim=300, output_dim=300, activation = 'relu', W_regularizer=l2(1e-5), b_regularizer=l2(1e-5)))
    model.add(Dropout(p2))
    model.add(Dense(input_dim=300, output_dim=300, activation = 'relu', W_regularizer=l2(1e-5), b_regularizer=l2(1e-5)))
    model.add(Dropout(p3))
    # model.add(Dense(input_dim=300, output_dim=300, activation = 'relu', W_regularizer=l2(1e-5), b_regularizer=l2(1e-5)))
    # model.add(Dropout(.2))
    model.add(Dense(input_dim=300, output_dim=2, activation = 'softmax', W_regularizer=l2(1e-5), b_regularizer=l2(1e-5)))
    return model

def cnn_optimise(W):
    # Number of feature maps (outputs of convolutional layer)
    N_fm = 300
    # kernel size of convolutional layer
    kernel_size = 8
    conv_input_width = W.shape[1]
    conv_input_height = 200     # maxlen of sentence

    model = Sequential()
    # Embedding layer (lookup table of trainable word vectors)
    model.add(Embedding(input_dim=W.shape[0], output_dim=W.shape[1], weights=[W], W_constraint=unitnorm(), init='uniform'))
    # Reshape word vectors from Embedding to tensor format suitable for Convolutional layer
    model.add(Reshape(dims=(1, conv_input_height, conv_input_width)))

    # first convolutional layer
    model.add(Convolution2D(nb_filter=N_fm, nb_row=kernel_size, nb_col=conv_input_width, border_mode='valid', W_regularizer=l2(0.0001), activation = 'relu'))
    # ReLU activation
    model.add(Dropout(0.5))

    # aggregate data in every feature map to scalar using MAX operation
    model.add(MaxPooling2D(pool_size=(conv_input_height-kernel_size+1, 1), border_mode='valid'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(output_dim=N_fm, activation = 'relu'))
    model.add(Dropout(0.5))
    # Inner Product layer (as in regular neural network, but without non-linear activation function)
    model.add(Dense(output_dim=2, activation = 'softmax'))
    # SoftMax activation; actually, Dense+SoftMax works as Multinomial Logistic Regression
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
    conv_input_height = 200     # maxlen of sentence

    cnn = Sequential()
    cnn.add(Embedding(input_dim = max_features, output_dim = 300, weights=[W]))
    cnn.add(Dropout(.5))
    cnn.add(Reshape(dims=(1, conv_input_height, conv_input_width)))
    # first convolutional layer
    cnn.add(Convolution2D(nb_filter=N_fm, nb_row=kernel_size, nb_col=conv_input_width, border_mode='valid', W_regularizer=l2(0.0001), activation = 'relu'))
    # ReLU activation
    cnn.add(Dropout(0.5))
    # aggregate data in every feature map to scalar using MAX operation
    cnn.add(MaxPooling2D(pool_size=(conv_input_height-kernel_size+1, 1), border_mode='valid'))
    cnn.add(Dropout(0.5))
    cnn.add(Flatten())
    cnn.add(Dense(output_dim=N_fm, activation = 'relu'))

    dan=Sequential()
    dan.add(Embedding(input_dim = max_features, output_dim = 300, weights=[W]))
    dan.add(Dropout(.5))
    dan.add(TimeDistributedMerge(mode='ave'))
    dan.add(Dense(input_dim=300, output_dim=300, activation = 'relu'))
    dan.add(Dropout(.5))
    dan.add(Dense(input_dim=300, output_dim=300, activation = 'relu'))
    dan.add(Dropout(.5))
    dan.add(Dense(input_dim=300, output_dim=300, activation = 'relu'))

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


def my_function(param1, param2, param3):
    ((x_train_idx_data, y_train_valence, y_train_labels,
     x_test_idx_data, y_test_valence, y_test_labels,
     x_valid_idx_data, y_valid_valence, y_valid_labels,
     x_train_polarity_idx_data, y_train_polarity,
     x_test_polarity_idx_data, y_test_polarity,
     x_valid_polarity_idx_data, y_valid_polarity), W) = build_keras_input()


    maxlen = 200  # cut texts after this number of words (among top max_features most common words)
    batch_size = 32
    (X_train, y_train), (X_test, y_test), (X_valid, y_valide) = (x_train_polarity_idx_data, y_train_polarity), (x_test_polarity_idx_data, y_test_polarity), (x_valid_polarity_idx_data, y_valid_polarity)
    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')
    # m= 0
    # for i in X_train:
    #     if len(i) >0:
    #         for j in i:
    #             if j > m:
    #                 m=j
    # print(m)
    max_features = W.shape[0] # shape of W: (13631, 300) , changed to 14027 through min_df = 3

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

    model = dan_pre_trained(W, param1,param2,param3)
    plot(model, to_file='./images/model.png')

    # try using different optimizers and different optimizer configs
    # model.compile(loss='binary_crossentropy', optimizer='adagrad', class_mode="binary")
    model.compile(loss='categorical_crossentropy', optimizer='adagrad')

    print("Train...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=100, validation_data=(X_test, y_test), show_accuracy=True, callbacks=[early_stopping])
    score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, show_accuracy=True)
    print('Test score:', score)
    print('Test accuracy:', acc)
    return acc

def result_analysis(filename):
    (param_grid, param_fitness) = load_pickle(filename)
    grid = ParameterGrid(param_grid)
    N=10         # top-n
    top_n_ind = np.argsort(param_fitness)[::-1][:N]         # top-n max value index

    for i in top_n_ind:
        print('Parameter setting: %s, acc: %s' % (str(list(grid)[i]), param_fitness[i]))


if __name__=='__main__':
    result_analysis('./tmp/grid_search_result.p')
    print('OK')
    exit()

    scope = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    param_grid = {'a':scope, 'b': scope, 'c': scope}
    param_fitness = []

    grid = ParameterGrid(param_grid)

    for params in grid:
        print('calculating... parameter: %s' % str(params))
        score = my_function(params['a'], params['b'], params['c'])
        print('Score: %s' % score)
        param_fitness.append(score)

    print('grid search complete.')
    # return the best fitness value and its settings
    best_fitness = np.min(np.array(param_fitness))
    best_ind = np.where(np.array(param_fitness)==best_fitness)[0]
    print('best fitness: %s' % best_fitness)
    print('best setting: %s' % str(list(grid)[best_ind]))

    dump_picle((param_grid, param_fitness), './tmp/grid_search_result.p')