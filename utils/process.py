import sys
sys.path.append('..')

import pandas as pd
import numpy as np
import os
import tqdm
import re
import string
import model

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from model import nn
import spacy
import utils
import torch
from torchtext import data
from torchtext import datasets

INPUT_DIM = 25002
EMBEDDING_DIM = 300
N_FILTERS = 100
FILTER_SIZES = [3, 4, 5]
OUTPUT_DIM = 1
DROPOUT = 0.5
PAD_IDX = 1


def process_content(string):
    """
    Process the content string data,
    if the string is nan (empty), replace it by ''
    else check if '#*;' in string, replace it by right symbol, 
    and if the meaning of '&#*;' is unknow, return the wrong info 
    
    Paras:
        string: the context
    
    Return:
        string: the modified string
        info: wrong info, if no wrong , then None
    """
    string = re.sub(r"'", '', string)
    string = re.sub(r"\n", '', string)
    i = 0
    # if len(string) != 0:
    while string[i] == ' ':
        i += 1
    string = string[i:]
    wrong_info = None 
    
    # print(re.sub(r'<br>', ' ', string))
    string = re.sub(r'<br>', ' ', string) # replace '<br>' by ' '
    string = re.sub(r'&#39;', "'", string) # replace '&#389' by "'"
    string = re.sub(r'&#8217;', "'", string) # replace '&#8217' by "'"
    string = re.sub(r'&#8230;', "...", string) # replace '&#8230' by '!'
    
    # find if there are others
    pattern = re.search(r'&#.*;', string)
    if pattern != None:
        wrong_info = 'there are some other wrong'
    # print(string)
    return string, wrong_info


def content2sentence(table, model_path):
    """
    Divide users' comment contents into separate sentence and 
    collect the postive or negtive scores
    """
    sentiment_model = model.CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)
    sentiment_model = sentiment_model.cuda()
    sentiment_model.load_state_dict(torch.load(model_path))
    nlp = spacy.load('en_core_web_lg')
    TEXT = data.Field(tokenize = 'spacy', batch_first = True, tokenizer_language='en_core_web_lg')
    LABEL = data.LabelField(dtype = torch.float)
    vdata, _ = datasets.IMDB.splits(TEXT, LABEL, root='.data/')
    TEXT.build_vocab(vdata, 
                 max_size = 25002, 
                 vectors = "glove.840B.300d",
                 vectors_cache = '.vector_cache/', 
                 unk_init = torch.Tensor.normal_)


    with tqdm.tqdm(total=len(table), desc='Remove empty content') as pbar:
        for i in range(len(table)):
            if len(table.Content[i]) == 0:
                table.drop(i, inplace=True)
            pbar.update(1)
    c2s = []
    with tqdm.tqdm(total=len(table), desc='Processing...') as pbar:
        for row in table.itertuples(index=False):
            date = row.ReviewData
            sentences = re.split('\!|\?|\.', process_content(row.Content)[0])[:-1]
            for sentence in sentences:
                sentence = sentence.lower()
                for c in string.punctuation:
                    sentence = sentence.replace(c, '')
                if len(sentence.split(' ')) >= 10:
                    emotion_label = utils.predict_sentiment(sentiment_model, sentence, nlp, TEXT, min_len=5)
                    c2s.append([row.id, sentence, emotion_label, date])
            pbar.update(1)
            
    return pd.DataFrame(c2s, columns=['cotent_id', 'sentence', 'emotion label', 'date'])


def count_classifier(words, pos, neg):
    """
    Count the positive or negative words in the sentence
    return the label
    """
    score = 0
    for word in words:
        if word + ' ' in pos:
            score += 1
        elif word + ' ' in neg:
            score -= 1
    return score

