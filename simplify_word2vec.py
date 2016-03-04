# coding: utf-8

# The purpose of this file is to simplify word2vec pre-trained models,
# as it is too big for loading it into an 8 GB memory laptop,
# and another reason is for fairly comparison because the words in GloVe vectors is much less than the
# vocabularies in word2vec models.
# The simplifying method is to just keep the vocabularies of word2vec, which appears in GloVe at the same time.


def save_word_vecs(wordVectors, outFileName):
    # the type of wordVectors is: dict()
    print('\nWriting down the vectors in ' + outFileName + '\n')
    outFile = open(outFileName, 'w', encoding='utf-8')
    for word, values in wordVectors.items():
        outFile.write(word + ' ')
        for val in wordVectors[word]:
            outFile.write('%.4f' % (val) + ' ')
        outFile.write('\n')
    outFile.close()


from load_data import load_embeddings

glove = load_embeddings("zh_tw", 'D:\Word_Embeddings\English\glove.6B\glove.6B.300d.txt')
vocabularies = glove.vocab.keys()
word2vec = load_embeddings('google_news', 'D:\Word_Embeddings\English\GoogleNews-vectors-negative300.bin')
common_keys = set(word2vec.keys()) & set(vocabularies)
wordVectors = dict()
for word in common_keys:
    wordVectors[word]=word2vec[word]
save_word_vecs(wordVectors, "D:\Word_Embeddings\English\simplified_word2vecs.txt")
