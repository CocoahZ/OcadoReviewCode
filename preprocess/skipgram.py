import gensim
import pickle
import os
import numpy as np
import argparse
import pandas as pd



# Class for a memory-friendly iterator over the dataset
class MySentences(object):
    def __init__(self, filepath):
        self.filepath = filepath
        docs = pd.read_table(self.filepath, names=['id', 'content_id', 'sentence', 'emotion label', 'date'])
        self.docs = docs.sentence
    def __iter__(self):
        for sentence in self.docs:
            yield sentence.split()
        

def main(args):
    # Gensim code to obtain the embeddings
    sentences = MySentences(args.data_file) # a memory-friendly iterator
    model = gensim.models.Word2Vec(sentences, min_count=args.min_count, sg=args.sg, size=args.dim_rho, 
        iter=args.iters, workers=args.workers, negative=args.negative_samples, window=args.window_size)

    # Write the embeddings to a file
    with open(args.emb_file, 'w', encoding='utf-8') as f:
        for v in list(model.wv.vocab):
            vec = list(model.wv.__getitem__(v))
            f.write(v + ' ')
            vec_str = ['%.9f' % val for val in vec]
            vec_str = " ".join(vec_str)
            f.write(vec_str + '\n')


if __name__ == '__main__':
    '''
    For a given docs, generate word embedding of each word by skipgram Word2vec model
    '''
    parser = argparse.ArgumentParser(description='The Embedded Topic Model')

    ### data and file related arguments
    parser.add_argument('--data_file', type=str, default='../results/processed_sentence.csv', help='a .csv file containing the corpus')
    parser.add_argument('--emb_file', type=str, default='../data/min_df_100/embeddings.txt', help='file to save the word embeddings')
    parser.add_argument('--dim_rho', type=int, default=300, help='dimensionality of the word embeddings')
    parser.add_argument('--min_count', type=int, default=2, help='minimum term frequency (to define the vocabulary)')
    parser.add_argument('--sg', type=int, default=1, help='whether to use skip-gram')
    parser.add_argument('--workers', type=int, default=25, help='number of CPU cores')
    parser.add_argument('--negative_samples', type=int, default=10, help='number of negative samples')
    parser.add_argument('--window_size', type=int, default=4, help='window size to determine context')
    parser.add_argument('--iters', type=int, default=50, help='number of iterationst')

    args = parser.parse_args()
    main(args)

