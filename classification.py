import numpy as np
from load_data import load_pickle

from sklearn.cross_validation import ShuffleSplit
from sklearn.linear_model import LogisticRegression

def build_data():
    positive_data = load_pickle('./tmp/amended_pos.p')
    negative_data = load_pickle('./tmp/amended_neg.p')
    X, Y = [], []
    for pos in positive_data.keys():
        X.append(positive_data[pos])
        Y.append(1)
    for neg in negative_data.keys():
        X.append(negative_data[neg])
        Y.append(0)
    return np.array(X), np.array(Y)

def train_model(X, Y):
    nub_iter = 20
    rs = ShuffleSplit(n=len(X), n_iter=nub_iter, test_size=0.2, indices=True, random_state=0)
    accuracy = []
    for train_index, test_index in rs:
        X_test, Y_test = X[test_index], Y[test_index]
        X_train, Y_train = X[train_index], Y[train_index]

        classifier = LogisticRegression()
        classifier.fit(X_train, Y_train)

        acc = classifier.score(X_test, Y_test)
        accuracy.append(acc)
        print('准确率Accuracy: %s.'%acc)
    print('平均准确率: %s.' % np.mean(np.array(accuracy)))


if __name__ == '__main__':
    X, Y = build_data()
    train_model(X, Y)
