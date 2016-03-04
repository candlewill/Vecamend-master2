from save_data import dump_picle
from load_data import load_Bing_Liu
from load_data import load_embeddings

def common_words(word_vectors, words_list):
    ####################################
    # for word2vec model:
    # full_words = word_vectors.vocab.keys()
    # for GloVe model:
    full_words = word_vectors.keys()
    ####################################
    same_words = set(words_list).intersection(full_words)
    print('Total Number: %s, same word number: %s.'%(len(words_list), len(same_words)))
    vector_dict=dict()
    for w in same_words:
        vector_dict[w]=word_vectors[w]
    dump_picle(vector_dict, './tmp/GloVe_common_negative_words.p')

def build_common_words_vectors(vecs='word2vec'):
    words_list = load_Bing_Liu('negative')
    word_vectors = load_embeddings('google_news', '/home/hs/Data/Word_Embeddings/google_news.bin')
    if vecs == 'GloVe':
        word_vectors = load_embeddings(arg='glove')
    common_words(word_vectors, words_list)

if __name__ == '__main__':
    build_common_words_vectors(vecs='GloVe')