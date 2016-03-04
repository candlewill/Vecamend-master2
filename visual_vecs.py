from load_data import load_pickle
from visualization import t_sne, draw_scatter
import numpy as np
import matplotlib.pyplot as plt


def visual_pos_neg_vecs(amended_pos_path='./tmp/amended_pos.p', amended_neg_path='./tmp/amended_neg.p'):
    amended_pos = load_pickle(amended_pos_path)
    amended_neg = load_pickle(amended_neg_path)
    nb_pos, nb_neg = len(amended_pos), len(amended_neg)
    print('There are %s positive words, and %s negative words.' % (nb_pos, nb_neg))
    num = 500
    vecs= [v for v in list(amended_pos.values())[:num]] + [v for v in list(amended_neg.values())[:num]]
    vecs = np.array(vecs)
    print('The shape of vecs is : %s row * %s columns.'%(vecs.shape))
    reduced_vecs = t_sne(vecs)
    print('The shape of reduced vecs is : %s row * %s columns.'%(reduced_vecs.shape))

    for i, vec in enumerate(vecs):
        if i < num:     # pos
            color = 'r'
        else:           # neg
            color = 'b'
        plt.plot(reduced_vecs[i,0], reduced_vecs[i,1], marker='o', color=color, markersize=8)
    plt.show()

def visual_original_vecs():
    pos_vectors_path = './tmp/common_positive_words.p'
    neg_vectors_path = './tmp/common_negative_words.p'
    visual_pos_neg_vecs(pos_vectors_path, neg_vectors_path)

if __name__ == '__main__':
    visual_pos_neg_vecs()
    visual_original_vecs()