from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from statistics import mean


# using t-sne to reduce dimension to 2, input: list of high-dim vectors, output: list of 2-D vectors.
def t_sne(vecs):
    ts = TSNE(2)
    reduced_vecs = ts.fit_transform(vecs)
    return reduced_vecs

def draw_scatter(x, y, x_labels, y_labels, title='CVAT 2.0 VA Scatter'):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(x, y, marker='o', color='#78A5A3')
    plt.axhline(mean(y), color='#CE5A57')
    plt.axvline(mean(x), color='#CE5A57')
    plt.xlabel(x_labels)
    plt.ylabel(y_labels)
    plt.title(title)
    plt.show()
    print('Figure displayed.')