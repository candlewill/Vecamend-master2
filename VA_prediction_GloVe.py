import numpy as np
from load_data import load_anew
from load_data import load_embeddings
from load_data import load_pickle
import os
from save_data import dump_picle
from sklearn.cross_validation import ShuffleSplit
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn import neighbors
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def check_same_words(words):
    model = load_embeddings(arg='zh_tw', filename="D:\Word_Embeddings\English\glove.6B\glove.6B.300d.txt")
    full_words = model.vocab.keys()
    same_words = set(words).intersection(full_words)
    print(set(words)-same_words)
    print(len(same_words))

def build_ori_anew_vectors(words):
    filename = './tmp/anew_vectors_glove.p'
    if os.path.isfile(filename):
        return load_pickle(filename)
    model = load_embeddings(arg='zh_tw', filename="D:\Word_Embeddings\English\glove.6B\glove.6B.300d.txt")
    vecs = []
    for w in words:
        vecs.append(model[w])
    vecs = np.array(vecs)
    dump_picle(vecs, filename)
    return vecs

def build_amended_anew_vectors(words):
    filename = './tmp/amended_anew_vectors_glove.p'
    if os.path.isfile(filename):
        return load_pickle(filename)
    model = load_embeddings(arg='zh_tw', filename="D:\Word_Embeddings\English\glove.6B\glove.6B.300d.txt")
    amended_pos = load_pickle('./tmp/amended_GloVe_pos.p')
    amended_neg = load_pickle('./tmp/amended_GloVe_neg.p')
    vecs = []
    for w in words:
        vec = amended_neg.get(w)
        if vec is None:
            vec = amended_pos.get(w)
        if vec is None:
            vec = model[w]
        vecs.append(vec)
    vecs = np.array(vecs)
    dump_picle(vecs, filename)
    return vecs

def regression(X, Y):
    nub_iter = 1
    ss = ShuffleSplit(X.shape[0], n_iter=nub_iter, test_size=0.2, random_state=0)

    for train_index, test_index in ss:
        X_test, Y_test = X[test_index], Y[test_index]
        X_train, Y_train = X[train_index], Y[train_index]

        ordinary_least_squares = linear_model.LinearRegression()
        ridge_regression = linear_model.Ridge(alpha=1)
        bayesian_regression = linear_model.BayesianRidge()
        svr = SVR(C=1.0, epsilon=0.2, kernel='linear')
        knn_reg = neighbors.KNeighborsRegressor(5, weights='distance')
        regrs = [ordinary_least_squares, ridge_regression, bayesian_regression, svr, knn_reg]

        for regr in regrs:
            regr.fit(X_train, Y_train)
            predict = regr.predict(X_test)
            np.seterr(invalid='ignore')

            true, pred = Y_test, predict
            MAE= mean_absolute_error(np.array(true), np.array(pred))
            MSE = mean_squared_error(np.array(true), np.array(pred))
            Pearson_r = pearsonr(np.array(true), np.array(pred))

            decimal = 4
            print('|%s|%s|%s|' % (round(MAE, decimal), round(MSE, decimal), round(Pearson_r[0], decimal)))

if __name__=='__main__':
    words, valence, arousal = load_anew('./resources/Lexicon/ANEW.txt')
    # remove_idx=[]       # the indeices
    # for i, w in enumerate(words):
    #     if w in {'glamour', 'skijump'}:
    #         remove_idx.append(i)
    #
    # for i in remove_idx[::-1]:
    #     words.pop(i)
    #     valence.pop(i)
    #     arousal.pop(i)
    # for i,j,k in zip(words, valence, arousal):
    #     print(i,j,k)
    vecs = build_amended_anew_vectors(words)
    print(vecs.shape)
    print("Valence")
    regression(vecs, np.array(valence))
    print("Arousal")
    regression(vecs, np.array(arousal))
