from load_data import load_sst
import numpy as np
from collections import defaultdict
from load_data import load_embeddings
from save_data import dump_picle
import os
from load_data import load_pickle


def clean_str(sent):
    sent = sent.strip().replace('.', '').replace(',', '')
    sent = sent.replace(';', '').replace('<br />', ' ')
    sent = sent.replace(':', '').replace('"', '')
    sent = sent.replace('(', '').replace(')', '')
    sent = sent.replace('!', '').replace('*', '')
    sent = sent.replace(' - ', ' ').replace(' -- ', '')
    sent = sent.replace('?', '')
    sent = sent.lower()
    return ' '.join(sent.split())


def get_vocab(corpus):
    vocab = defaultdict(int)
    for sent in corpus:
        for word in clean_str(sent).split():
            vocab[word] += 1
    print('The total number of vocabulary is: %s. ' % len(vocab))
    return vocab

def add_unknown_words(word_vecs, vocab, min_df=3, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)
    return word_vecs

# word_vecs is the model of word2vec
def build_embedding_matrix(word_vecs, vocab, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    union = (set(word_vecs.keys()) & set(vocab.keys()))
    vocab_size = len(union)
    print('The number of words occuring in corpus and word2vec simutaneously: %s.' % vocab_size)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size + 1, k))
    W[0] = np.zeros(k, dtype=np.float32)
    for i, word in enumerate(union, start=1):
        print(word, i)
        W[i] = word_vecs[word]
        word_idx_map[word] = i  # dict
    return W, word_idx_map


def sent2ind(sent, word_idx_map):
    """
    Transforms sentence into a list of indices.
    """
    x = []
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    return x


def make_idx_data(sentences, word_idx_map):
    """
    Transforms sentences (corpus, a list of sentence) into a 2-d matrix.
    """
    idx_data = []
    for sent in sentences:
        idx_sent = sent2ind(clean_str(sent), word_idx_map)
        idx_data.append(idx_sent)
    # idx_data = np.array(idx_data, dtype=np.int)
    return idx_data


def build_keras_input():
    filename_data, filename_w = './tmp/indexed_data.p', './tmp/Weight.p'

    if os.path.isfile(filename_data) and os.path.isfile(filename_w):
        data = load_pickle(filename_data)
        W = load_pickle(filename_w)
        print('Load OK.')
        return (data, W)

    # load data from pickle
    (x_train, y_train_valence, y_train_labels,
     x_test, y_test_valence, y_test_labels,
     x_valid, y_valid_valence, y_valid_labels,
     x_train_polarity, y_train_polarity,
     x_test_polarity, y_test_polarity,
     x_valid_polarity, y_valid_polarity) = load_sst(path='./resources/stanfordSentimentTreebank/')

    vocab = get_vocab(x_train)
    # word_vecs = load_embeddings('google_news', '/home/hs/Data/Word_Embeddings/google_news.bin')
    word_vecs = load_embeddings('glove')
    word_vecs = add_unknown_words(word_vecs, vocab)
    W, word_idx_map = build_embedding_matrix(word_vecs, vocab)

    x_train_idx_data = make_idx_data(x_train, word_idx_map)
    x_test_idx_data = make_idx_data(x_test, word_idx_map)
    x_valid_idx_data = make_idx_data(x_valid, word_idx_map)
    x_train_polarity_idx_data = make_idx_data(x_train_polarity, word_idx_map)
    x_test_polarity_idx_data = make_idx_data(x_test_polarity, word_idx_map)
    x_valid_polarity_idx_data = make_idx_data(x_valid_polarity, word_idx_map)

    data = (x_train_idx_data, y_train_valence, y_train_labels,
            x_test_idx_data, y_test_valence, y_test_labels,
            x_valid_idx_data, y_valid_valence, y_valid_labels,
            x_train_polarity_idx_data, y_train_polarity,
            x_test_polarity_idx_data, y_test_polarity,
            x_valid_polarity_idx_data, y_valid_polarity)

    dump_picle(data, filename_data)
    dump_picle(W, filename_w)
    return (data, W)

def build_keras_input_amended():
    filename_data, filename_w = './tmp/amended_indexed_data.p', './tmp/amended_Weight.p'

    if os.path.isfile(filename_data) and os.path.isfile(filename_w):
        data = load_pickle(filename_data)
        W = load_pickle(filename_w)
        print('Load OK.')
        return (data, W)

    # load data from pickle
    (x_train, y_train_valence, y_train_labels,
     x_test, y_test_valence, y_test_labels,
     x_valid, y_valid_valence, y_valid_labels,
     x_train_polarity, y_train_polarity,
     x_test_polarity, y_test_polarity,
     x_valid_polarity, y_valid_polarity) = load_sst(path='./resources/stanfordSentimentTreebank/')

    vocab = get_vocab(x_train)
    # word_vecs = load_embeddings('google_news', '/home/hs/Data/Word_Embeddings/google_news.bin')
    # word_vecs = load_embeddings('glove')

    # load amended word vectors
    word_vecs = load_embeddings('amended_word2vec')

    word_vecs = add_unknown_words(word_vecs, vocab)
    W, word_idx_map = build_embedding_matrix(word_vecs, vocab)

    x_train_idx_data = make_idx_data(x_train, word_idx_map)
    x_test_idx_data = make_idx_data(x_test, word_idx_map)
    x_valid_idx_data = make_idx_data(x_valid, word_idx_map)
    x_train_polarity_idx_data = make_idx_data(x_train_polarity, word_idx_map)
    x_test_polarity_idx_data = make_idx_data(x_test_polarity, word_idx_map)
    x_valid_polarity_idx_data = make_idx_data(x_valid_polarity, word_idx_map)

    data = (x_train_idx_data, y_train_valence, y_train_labels,
            x_test_idx_data, y_test_valence, y_test_labels,
            x_valid_idx_data, y_valid_valence, y_valid_labels,
            x_train_polarity_idx_data, y_train_polarity,
            x_test_polarity_idx_data, y_test_polarity,
            x_valid_polarity_idx_data, y_valid_polarity)

    dump_picle(data, filename_data)
    dump_picle(W, filename_w)
    return (data, W)

def keras_nn_input(word_vectors_model, amending):
    if word_vectors_model == 'word2vec':
        if amending == True:
            filename_data, filename_w = './tmp/amended_w2v_indexed_data.p', './tmp/amended_w2v_Weight.p'
        elif amending == False:
            filename_data, filename_w = './tmp/w2v_indexed_data.p', './tmp/w2v_Weight.p'
        else:
            raise Exception('Wrong!')
    elif word_vectors_model == 'GloVe':
        if amending == True:
            filename_data, filename_w = './tmp/amended_GloVe_indexed_data.p', './tmp/amended_GloVe_Weight.p'
        elif amending == False:
            filename_data, filename_w = './tmp/GloVe_indexed_data.p', './tmp/GloVe_Weight.p'
        else:
            raise Exception('Wrong!')

    elif word_vectors_model == 'retrofitted_GloVe':
        filename_data, filename_w = './tmp/retrofitted_GloVe_indexed_data.p', './tmp/retrofitted_GloVe_Weight.p'
    elif word_vectors_model == 'retrofitted_word2vec':
        filename_data, filename_w = './tmp/retrofitted_word2vec_indexed_data.p', './tmp/retrofitted_word2vec_Weight.p'

    else:
        raise Exception('Wrong parameter!')

    if os.path.isfile(filename_data) and os.path.isfile(filename_w):
        data = load_pickle(filename_data)
        W = load_pickle(filename_w)
        print('Load OK, parameters: word_vectors_model = %s, amending = %s'%(word_vectors_model, amending))
        return (data, W)

    # load data from pickle
    (x_train, y_train_valence, y_train_labels,
     x_test, y_test_valence, y_test_labels,
     x_valid, y_valid_valence, y_valid_labels,
     x_train_polarity, y_train_polarity,
     x_test_polarity, y_test_polarity,
     x_valid_polarity, y_valid_polarity) = load_sst(path='./resources/stanfordSentimentTreebank/')

    vocab = get_vocab(x_train)

    if word_vectors_model == 'word2vec':
        if amending == True:
            word_vecs = load_embeddings('amended_word2vec')
        elif amending == False:
            word_vecs = load_embeddings('google_news', '/home/hs/Data/Word_Embeddings/google_news.bin')
        else:
            raise Exception('Wrong!')
    elif word_vectors_model == 'GloVe':
        if amending == True:
            word_vecs = load_embeddings('amended_glove')
        elif amending == False:
            word_vecs = load_embeddings('glove')
        else:
            raise Exception('Wrong!')
    elif word_vectors_model == 'retrofitted_GloVe':
        word_vecs = load_embeddings('zh_tw', 'D:\Word_Embeddings\English\glove.6B\GloVe_out_vec_file.txt')
        # convert gensim model to dict type
        w2v = dict()
        for key in word_vecs.vocab.keys():
            w2v[key] = word_vecs[key]
        word_vecs = w2v

    elif word_vectors_model == 'retrofitted_word2vec':
        word_vecs = load_embeddings('zh_tw', 'D:\Word_Embeddings\English\word2vec_out_vec_file.txt')
        # convert gensim model to dict type
        w2v = dict()
        for key in word_vecs.vocab.keys():
            w2v[key] = word_vecs[key]
        word_vecs = w2v

    else:
        raise Exception('Wrong parameter!')

    word_vecs = add_unknown_words(word_vecs, vocab)
    W, word_idx_map = build_embedding_matrix(word_vecs, vocab)

    x_train_idx_data = make_idx_data(x_train, word_idx_map)
    x_test_idx_data = make_idx_data(x_test, word_idx_map)
    x_valid_idx_data = make_idx_data(x_valid, word_idx_map)
    x_train_polarity_idx_data = make_idx_data(x_train_polarity, word_idx_map)
    x_test_polarity_idx_data = make_idx_data(x_test_polarity, word_idx_map)
    x_valid_polarity_idx_data = make_idx_data(x_valid_polarity, word_idx_map)

    data = (x_train_idx_data, y_train_valence, y_train_labels,
            x_test_idx_data, y_test_valence, y_test_labels,
            x_valid_idx_data, y_valid_valence, y_valid_labels,
            x_train_polarity_idx_data, y_train_polarity,
            x_test_polarity_idx_data, y_test_polarity,
            x_valid_polarity_idx_data, y_valid_polarity)

    dump_picle(data, filename_data)
    dump_picle(W, filename_w)
    print('Load OK, parameters: word_vectors_model = %s, amending = %s'%(word_vectors_model, amending))
    return (data, W)


def nn_input(word_vectors_model = 'word2vec', amending = False):

    ###################################### Hyper-parameters #######################################
    word_vectors_model = 'word2vec'      # values: 'word2vec','GloVe', 'retrofitted_GloVe', 'retrofitted_word2vec'
    amending = False                         # values: True, False
    ###############################################################################################
    data = keras_nn_input(word_vectors_model, amending)
    return data

if __name__ == '__main__':
    # ((x_train_idx_data, y_train_valence, y_train_labels,
    #  x_test_idx_data, y_test_valence, y_test_labels,
    #  x_valid_idx_data, y_valid_valence, y_valid_labels,
    #  x_train_polarity_idx_data, y_train_polarity,
    #  x_test_polarity_idx_data, y_test_polarity,
    # x_valid_polarity_idx_data, y_valid_polarity), W) = build_keras_input()
    # build_keras_input_amended()         # using amended word2vec
    nn_input()



