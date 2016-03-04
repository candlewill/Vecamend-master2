import csv
import os
import pickle
import gensim
from gensim.models import Doc2Vec
import re
from save_data import dump_picle
import numpy as np


def load_anew(filepath=None):
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        words, arousal, valence = [], [], []
        for line in reader:
            words.append(line[0])
            valence.append(float(line[1]))
            arousal.append(float(line[2]))
    return words, valence, arousal


def load_extend_anew(D=False):
    print('Loading extend_anew lexicon')
    with open('./resource/extend_ANEW.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        words, arousal, valence, dominance = [], [], [], []
        for line in reader:
            if reader.line_num == 1:
                continue
            words.append(line[1])
            arousal.append(float(line[5]))
            valence.append(float(line[2]))
            if D == True:
                dominance.append(float(line[8]))
    print('Loading extend_anew lexicon complete')
    if D == True:
        return words, valence, arousal, dominance
    else:
        return words, valence, arousal


def load_Bing_Liu(polarity=None):
    if polarity == 'positive':
        filename = './resources/Lexicon/Bing_Liu/positive-words.txt'
    elif polarity == 'negative':
        filename = './resources/Lexicon/Bing_Liu/negative-words.txt'
    else:
        raise Exception('Wrong Argument.')
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        words = []
        for line in reader:
            words.append(line[0])
            # print(line[0])
    return words


def load_csv(filename):
    out = []
    with open(filename, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f, quoting=csv.QUOTE_NONE)
        for line in reader:
            out.append(line)
    return out


def load_sentiment140(filename):
    print('start loading data...')
    # ��ʽ��"4","1467822272","Mon Apr 06 22:22:45 PDT 2009","NO_QUERY","ersle","I LOVE @Health4UandPets u guys r the best!! "
    with open(filename, 'rt', encoding='ISO-8859-1') as f:
        inpTweets = csv.reader(f, delimiter=',', quotechar='"')
        X = []  # sentiment
        Y = []  # tweets
        for row in inpTweets:
            sentiment = (1 if row[0] == '4' else 0)
            tweet = row[5]
            X.append(sentiment)
            Y.append(tweet)
        # end loop
        return Y, X


def load_vader(filename):
    with open(filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        texts, ratings = [], []
        for line in reader:
            texts.append(line[2])
            ratings.append(float(line[1]))
    return texts, ratings


def load_embeddings(arg=None, filename='None'):
    if arg == 'zh_tw':  # dim = 400
        model = gensim.models.Word2Vec.load_word2vec_format(filename, binary=False)
    elif arg == 'google_news':
        model = gensim.models.Word2Vec.load_word2vec_format(filename, binary=True)  # C binary format
        w2v = dict()
        for key in model.vocab.keys():
            w2v[key] = model[key]
        return w2v
    elif arg == 'glove':
        glove_path = 'D:\Word_Embeddings\English\glove.6B\glove.6B.300d.txt'
        model = dict()
        with open(glove_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f, quoting=csv.QUOTE_NONE, delimiter=' ')
            for line in reader:
                # print(line[0])
                # split = line.split()
                try:
                    model[line[0]] = np.array(line[1:], dtype=float)
                except:
                    pass
        return model
    elif arg == 'amended_word2vec':
        model = gensim.models.Word2Vec.load_word2vec_format('/home/hs/Data/Word_Embeddings/google_news.bin',
                                                            binary=True)  # C binary format
        w2v = dict()
        for key in model.vocab.keys():
            w2v[key] = model[key]
        pos_vecs = load_pickle('./tmp/amended_pos.p')
        neg_vecs = load_pickle('./tmp/amended_neg.p')
        for p in pos_vecs:
            w2v[p] = pos_vecs[p]
        for n in neg_vecs:
            w2v[n] = neg_vecs[n]
        print('Using amended word vectors.')
        return w2v

    elif arg == 'amended_glove':
        glove_path = '/home/hs/Data/Word_Embeddings/glove.840B.300d.txt'
        model = dict()
        with open(glove_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f, quoting=csv.QUOTE_NONE, delimiter=' ')
            for line in reader:
                # print(line[0])
                # split = line.split()
                try:
                    model[line[0]] = np.array(line[1:], dtype=float)
                except:
                    pass
        pos_vecs = load_pickle('./tmp/amended_GloVe_pos.p')
        neg_vecs = load_pickle('./tmp/amended_GloVe_neg.p')
        for p in pos_vecs:
            model[p] = pos_vecs[p]
        for n in neg_vecs:
            model[n] = neg_vecs[n]
        print('Using amended word vectors.')
        return model

    elif arg == 'CVAT':  # dim = 50
        model = gensim.models.Word2Vec.load(None)
    elif arg == 'twitter':  # dim = 50
        model = Doc2Vec.load('./data/acc/docvecs_twitter.d2v')
    else:
        raise Exception('Wrong Argument.')
    print('Load Model Complete.')
    return model


def load_pickle(filename):
    out = pickle.load(open(filename, "rb"))
    return out


def load_sst(path=None, level=None):
    filename = './tmp/SST.p'
    if os.path.isfile(filename):
        print('Load OK.')
        return load_pickle(filename)

    def cleanStr(string):
        string = re.sub(r'^A-Za-z0-9(),!?\'\`', ' ', string)
        string = re.sub(r'\s{2,}', ' ', string)
        string = string.replace('Ã¡', 'á').replace('Ã©', 'é').replace('Ã±', 'ñ').replace('Â', '').replace('Ã¯', 'ï')
        string = string.replace('Ã¼', 'ü').replace('Ã¢', 'â').replace('Ã¨', 'è').replace('Ã¶', 'ö').replace('Ã¦', 'æ')
        string = string.replace('Ã³', 'ó').replace('Ã»', 'û').replace('Ã´', 'ô').replace('Ã£', 'ã').replace('Ã§', 'ç')
        string = string.replace('Ã ', 'à ').replace('Ã', 'í').replace('í­', 'í')
        return string

    # sentiment label
    sentiment_file = open(path + 'sentiment_labels.txt', 'r')
    sentiment_label = {}
    n = 0
    for line in sentiment_file:
        lines = line.strip().split('|')
        if n > 0:
            sentiment_label[int(lines[0])] = float(lines[1])
        n += 1
    sentiment_file.close()

    # phrase dict
    dict_file = open(path + 'dictionary.txt', 'r')
    phrase_dict = {}
    for line in dict_file:
        # line = line.decode('utf-8')
        lines = line.strip().split('|')
        phrase_dict[lines[0]] = int(lines[1])
    dict_file.close()

    # sentence dict
    sentence_file = open(path + 'datasetSentences.txt', 'r')
    sentence_dict = {}
    n = 0
    for line in sentence_file:
        # line = line.decode('utf-8')
        line = line.replace('-LRB-', '(')
        line = line.replace('-RRB-', ')')
        lines = line.strip().split('\t')
        if n > 0:
            sentence_dict[int(lines[0])] = lines[1]
        n += 1
    sentence_file.close()

    # datasplit
    datasplit_file = open(path + 'datasetSplit.txt', 'r')
    split_dict = {}
    n = 0
    for line in datasplit_file:
        lines = line.strip().split(',')
        if n > 0:
            split_dict[int(lines[0])] = int(lines[1])
        n += 1
    datasplit_file.close()

    size = len(sentence_dict)  # size = 11855
    # for i in range(1000):
    #     senti = sentiment_label[phrase_dict[cleanStr(sentence_dict[i + 1])]]
    #     print(i, senti, cleanStr(sentence_dict[i + 1]))
    # exit()
    x_train, y_train_valence, y_train_labels = [], [], []
    x_test, y_test_valence, y_test_labels = [], [], []
    x_valid, y_valid_valence, y_valid_labels = [], [], []

    x_train_polarity, y_train_polarity = [], []
    x_test_polarity, y_test_polarity = [], []
    x_valid_polarity, y_valid_polarity = [], []

    for i in range(size):
        # print sentence_dict[i+1].encode('utf-8')
        sentence = cleanStr(sentence_dict[i + 1])
        senti = sentiment_label[phrase_dict[sentence]]

        # print(senti, sentence)
        labels, polarity = None, None
        if 0 <= senti <= 0.2:
            labels = 1
            polarity = 0
        if 0.2 < senti <= 0.4:
            labels = 2
            polarity = 0
        if 0.4 < senti <= 0.6:
            labels = 3
        if 0.6 < senti <= 0.8:
            labels = 4
            polarity = 1
        if 0.8 < senti <= 1:
            labels = 5
            polarity = 1
        if labels is None:
            raise Exception('Sentiment Error !')

        if split_dict[i + 1] == 1:
            x_train.append(sentence)
            y_train_valence.append(senti)
            y_train_labels.append(labels)
            if polarity is not None:
                x_train_polarity.append(sentence)
                y_train_polarity.append(polarity)
        elif split_dict[i + 1] == 2:
            x_test.append(sentence)
            y_test_valence.append(senti)
            y_test_labels.append(labels)
            if polarity is not None:
                x_test_polarity.append(sentence)
                y_test_polarity.append(polarity)
        else:
            x_valid.append(sentence)
            y_valid_valence.append(senti)
            y_valid_labels.append(labels)
            if polarity is not None:
                x_valid_polarity.append(sentence)
                y_valid_polarity.append(polarity)

    print("Fine-grained: #training: %s, #valid: %s, #test: %s" % (len(x_train), len(x_valid), len(x_test)))
    print("Binary classification: #train: %s, #valid: %s, #test: %s" % (
        len(x_train_polarity), len(x_valid_polarity), len(x_test_polarity)))

    # t = zip(x_train, y_train)
    # random.shuffle(t)
    # x_train, y_train = zip(*t)
    output = (x_train, y_train_valence, y_train_labels,
              x_test, y_test_valence, y_test_labels,
              x_valid, y_valid_valence, y_valid_labels,
              x_train_polarity, y_train_polarity,
              x_test_polarity, y_test_polarity,
              x_valid_polarity, y_valid_polarity)
    dump_picle(output, filename)
    print('Data saved and load successfully.')
    return output
