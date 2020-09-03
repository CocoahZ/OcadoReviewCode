import pandas as pd
import numpy as np
import os
import tqdm
import re
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

import spacy
import torch
from torch.nn import functional as F


def read_pd_xls(path):
    """
    Read the xls file to pandas DataFrames 
    And use 0 or '' to replace the None data 

    Input:
        path: the single xls file path or a list of path

    Return:
        the merged pandas DataFrames
    """
    if not isinstance(path, list):
        path = list(path)
    table = []
    for i in path:
        table.append(pd.read_excel(i, names=['id',
                                             'ReviewCount',
                                             'Title',
                                             'Content',
                                             'ReviewData',
                                             'ReviewTime',
                                             'ReviewrName',
                                             'Comment',
                                             'CommentDate',
                                             'CommentTime',
                                             'PageNumber']))
    ans = table[0]
    for i in range(1, len(table)):
        ans = ans.append(table[i], ignore_index=True)
    ans.ReviewCount.fillna(0, inplace=True)
    ans.Title.fillna('', inplace=True)
    ans.Content.fillna('', inplace=True)
    ans.ReviewrName.fillna('', inplace=True)
    ans.Comment.fillna('', inplace=True)
    ans.CommentDate.fillna('', inplace=True)
    ans.CommentTime.fillna('', inplace=True)
    return ans


def load_pos_neg_wordsbag(pos_path, neg_path):
    pos = open(pos_path, encoding='unicode_escape').read().splitlines()
    neg = open(neg_path, encoding='unicode_escape').read().splitlines()
    return pos, neg


def normalize(feature):
    for i in  range(len(feature)):
        sum_ = sum(feature[i])
        for j in range(len(feature[i])):
            if feature[i][j] > 0:
                feature[i][j] /= sum_
    return feature

def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
 
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.scatter(data[i, 0], data[i, 1], color=plt.cm.Set1(label[i] / 10.))
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

def predict_sentiment(model, sentence, nlp, TEXT, min_len = 5):
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to('cuda')
    tensor = tensor.unsqueeze(0)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()

def get_topic_diversity(beta, topk):
    num_topics = beta.shape[0]
    list_w = np.zeros((num_topics, topk))
    for k in range(num_topics):
        idx = beta[k,:].argsort()[-topk:][::-1]
        list_w[k,:] = idx
    n_unique = len(np.unique(list_w))
    TD = n_unique / (topk * num_topics)
    print('Topic diveristy is: {}'.format(TD))

def get_document_frequency(data, wi, wj=None):
    if wj is None:
        D_wi = 0
        for l in range(len(data)):
            doc = data[l].squeeze(0)
            if len(doc) == 1: 
                continue
            else:
                doc = doc.squeeze()
            if wi in doc:
                D_wi += 1
        return D_wi
    D_wj = 0
    D_wi_wj = 0
    for l in range(len(data)):
        doc = data[l].squeeze(0)
        if len(doc) == 1: 
            doc = [doc.squeeze()]
        else:
            doc = doc.squeeze()
        if wj in doc:
            D_wj += 1
            if wi in doc:
                D_wi_wj += 1
    return D_wj, D_wi_wj 

def get_topic_coherence(beta, data, vocab):
    D = len(data) ## number of docs...data is list of documents
    print('D: ', D)
    TC = []
    num_topics = len(beta)
    for k in range(num_topics):
        print('k: {}/{}'.format(k, num_topics))
        top_10 = list(beta[k].argsort()[-11:][::-1])
        top_words = [vocab[a] for a in top_10]
        TC_k = 0
        counter = 0
        for i, word in enumerate(top_10):
            # get D(w_i)
            D_wi = get_document_frequency(data, word)
            j = i + 1
            tmp = 0
            while j < len(top_10) and j > i:
                # get D(w_j) and D(w_i, w_j)
                D_wj, D_wi_wj = get_document_frequency(data, word, top_10[j])
                # get f(w_i, w_j)
                if D_wi_wj == 0:
                    f_wi_wj = -1
                else:
                    f_wi_wj = -1 + ( np.log(D_wi) + np.log(D_wj)  - 2.0 * np.log(D) ) / ( np.log(D_wi_wj) - np.log(D) )
                # update tmp: 
                tmp += f_wi_wj
                j += 1
                counter += 1
            # update TC_k
            TC_k += tmp 
        TC.append(TC_k)
    print('counter: ', counter)
    print('num topics: ', len(TC))
    TC = np.mean(TC) / counter
    print('Topic coherence is: {}'.format(TC))

def nearest_neighbors(word, embeddings, vocab):
    vectors = embeddings.data.cpu().numpy() 
    index = vocab.index(word)
    print('vectors: ', vectors.shape)
    query = vectors[index]
    print('query: ', query.shape)
    ranks = vectors.dot(query).squeeze()
    denom = query.T.dot(query).squeeze()
    denom = denom * np.sum(vectors**2, 1)
    denom = np.sqrt(denom)
    ranks = ranks / denom
    mostSimilar = []
    [mostSimilar.append(idx) for idx in ranks.argsort()[::-1]]
    nearest_neighbors = mostSimilar[:20]
    nearest_neighbors = [vocab[comp] for comp in nearest_neighbors]
    return nearest_neighbors
