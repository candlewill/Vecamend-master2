from __future__ import absolute_import
from __future__ import print_function

import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from sentiment_classification import nn_input
from keras.layers.core import Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.callbacks import EarlyStopping
from keras.layers.core import Activation

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support


def classify_metrics(predict, true):
    f1 = f1_score(true, predict, average='binary')
    precision_binary, recall_binary, fbeta_score_binary, _ = precision_recall_fscore_support(true, predict,
                                                                                             average='binary')
    accuracy = accuracy_score(true, predict)
    print('正确率(Accuracy)：%.3f\nF值(Macro-F score)：%.3f' % (accuracy, f1))
    print('精确度(Precision):%.3f\n召回率：%.3f\nF值: %.3f' % (precision_binary, recall_binary, fbeta_score_binary))

def cnn(W):
    nb_filter = 250
    filter_length = 3
    hidden_dims = 250

    maxlen = 200

    model = Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(W.shape[0], W.shape[1], input_length=maxlen, weights=[W]))
    model.add(Dropout(0.25))

    # we add a Convolution1D, which will learn nb_filter
    # word group filters of size filter_length:
    model.add(Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1))
    # we use standard max pooling (halving the output of the previous layer):
    model.add(MaxPooling1D(pool_length=2))

    # We flatten the output of the conv layer,
    # so that we can add a vanilla dense layer:
    model.add(Flatten())

    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.25))
    model.add(Activation('relu'))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(2))
    model.add(Activation('softmax'))

    return model


if __name__ == "__main__":
    ((x_train_idx_data, y_train_valence, y_train_labels,
      x_test_idx_data, y_test_valence, y_test_labels,
      x_valid_idx_data, y_valid_valence, y_valid_labels,
      x_train_polarity_idx_data, y_train_polarity,
      x_test_polarity_idx_data, y_test_polarity,
      x_valid_polarity_idx_data, y_valid_polarity), W) = nn_input()  # build_keras_input_amended or build_keras_input

    maxlen = 200  # cut texts after this number of words (among top max_features most common words)
    batch_size = 8
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
    old_y_test = y_test
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    y_valide = np_utils.to_categorical(y_valide, nb_classes)

    model = cnn(W)

    # try using different optimizers and different optimizer configs
    # model.compile(loss='binary_crossentropy', optimizer='adagrad', class_mode="binary")
    model.compile(loss='categorical_crossentropy', optimizer='adagrad')  # adagrad

    print("Train...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=2, validation_data=(X_test, y_test),
              show_accuracy=True,
              callbacks=[early_stopping])
    score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, show_accuracy=True)
    print('Test score:', score)
    print('Test accuracy:', acc)
    predict = model.predict_classes(X_test, batch_size=batch_size).reshape((1, len(X_test)))[0]
    print('Y_test: %s' %str(old_y_test))
    print('Predict value: %s' % str(predict))

    classify_metrics(predict, old_y_test)
