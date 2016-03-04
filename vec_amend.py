import numpy as np
from copy import deepcopy
from load_data import load_pickle
from draw import draw_line_chart
from save_data import dump_picle

def get_centroid(vectors):
    length = len(vectors)
    size = vectors[list(vectors.keys())[0]].shape[1]
    vec = np.zeros(size).reshape((1, size))
    for v in vectors:
        # print(vec.shape, vectors[v].reshape((1, size)).shape)
        vec += vectors[v].reshape((1, size))
    return vec/length

def euclidean_dist(P,Q):
    return np.linalg.norm(P-Q)

def cost_function(ori_pos,ori_neg,pos_vectors, neg_vectors, alpha, beta, gamma):
    A, B, C =0,0,0
    pos_centroid = get_centroid(pos_vectors)
    neg_centroid = get_centroid(neg_vectors)
    for p in pos_vectors:
        A += alpha*euclidean_dist(pos_vectors[p], pos_centroid)**2-beta*euclidean_dist(pos_vectors[p],neg_centroid)**2
        C += gamma*euclidean_dist(pos_vectors[p], ori_pos[p])
    for n in neg_vectors:
        B += alpha*euclidean_dist(neg_vectors[n], neg_centroid)**2-beta*euclidean_dist(neg_vectors[n],pos_centroid)**2
        C += gamma*euclidean_dist(neg_vectors[n], ori_neg[n])
    return (A+B+C)

def amend(pos_vectors, neg_vectors):
    alpha, beta, gamma = 1, 0.3, 1
    amended_pos, amended_neg = deepcopy(pos_vectors), deepcopy(neg_vectors)
    cfs = []
    nb_iter = 30
    for it in range(nb_iter):
        cf = cost_function(pos_vectors,neg_vectors,amended_pos, amended_neg, alpha, beta, gamma)
        cfs.append(cf)
        print('Cost function: %s'%cf)
        print('iterative %s is starting.' % it)
        pos_centroid = get_centroid(amended_pos)
        neg_centroid = get_centroid(amended_neg)
        tmp_pos_dict = dict()
        for w in pos_vectors:
            tmp_pos_dict[w]=(alpha*pos_centroid-beta*neg_centroid+gamma*pos_vectors[w])/(alpha-beta+gamma)
        tmp_neg_dict = dict()
        for w in neg_vectors:
            tmp_neg_dict[w]=(alpha*neg_centroid-beta*pos_centroid+gamma*neg_vectors[w])/(alpha-beta+gamma)
        amended_pos = deepcopy(tmp_pos_dict)
        amended_neg = deepcopy(tmp_neg_dict)
        print('iterative %s is completed.' % it)
    draw_line_chart(range(nb_iter), cfs, 'Iterative', 'Cost function')

    # To keep the data type the same
    for x in amended_pos:
        amended_pos[x] = amended_pos[x].tolist()[0]
        # print(amended_pos[x])
    for y in amended_neg:
        amended_neg[y] = amended_neg[y].tolist()[0]

    return amended_pos, amended_neg

def build_amended_vectors(arg='word2vec'):
    prefix = None if arg == 'word2vec' else 'GloVe_'
    pos_vectors = load_pickle('./tmp/'+prefix+'common_positive_words.p')
    neg_vectors = load_pickle('./tmp/'+prefix+'common_negative_words.p')
    size = len(pos_vectors[list(pos_vectors.keys())[0]])
    print('The dimension of word vectors: %s.' % size)
    for k in pos_vectors:
        pos_vectors[k]=np.array(pos_vectors[k]).reshape((1, size))
    for k in neg_vectors:
        neg_vectors[k]=np.array(neg_vectors[k]).reshape((1, size))
    amended_pos, amended_neg = amend(pos_vectors, neg_vectors)
    dump_picle(amended_pos, './tmp/amended_'+prefix+'pos.p')
    dump_picle(amended_neg, './tmp/amended_'+prefix+'neg.p')

if __name__=='__main__':
    build_amended_vectors(arg='GloVe')           # arg values: word2vec, GloVe
