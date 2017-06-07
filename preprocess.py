# Process data for experiments
# Data Source:
# 1 http://qwone.com/~jason/20Newsgroups/ 
# 2 http://www.cs.jhu.edu/~mdredze/datasets/sentiment/ 
# 3 http://www.cs.cornell.edu/~ainur/data/sle_sent_movieReviews.tar.gz
# 4 http://www.cs.cornell.edu/~ainur/data/sle_sent_convote.tar.gz

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import urllib.request
import tarfile
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import vstack
from nltk import sent_tokenize
import collections
import gensim
import numpy as np
import pickle
import random
import scipy as sp

def download_data(data, data_path):

    full_path = os.path.join(data_path, data)

    if data == 'processed_stars.tar.gz':
        urllib.request.urlretrieve('http://www.cs.jhu.edu/~mdredze/datasets/sentiment/processed_stars.tar.gz', 
                full_path)

    elif data == 'sle_sent_movieReviews.tar.gz':
        urllib.request.urlretrieve('http://www.cs.cornell.edu/~ainur/data/sle_sent_movieReviews.tar.gz', 
                full_path)

    elif data == 'sle_sent_convote.tar.gz':
        urllib.request.urlretrieve('http://www.cs.cornell.edu/~ainur/data/sle_sent_convote.tar.gz', 
                full_path)

    tar = tarfile.open(full_path)
    tar.extractall(data_path)
    tar.close()
            

def process_stars():
    pass

def process_movies(data, data_path):
    """
    Process moview dataset
    Output:
        list of list of array with shape [number_of_documents, number_of_sentences, embedding_size]
    """

    dat = {'train': {'data':[], 'tar':[]}, 'test': {'data':[], 'tar':[]}}
    file_dict = {}
    train_val_file_list = []
    test_file_list = []

    #Use CVList, leave 
    for (dirpath, dirnames, filenames) in os.walk(data_path):
        if data.split('.')[0] in dirpath:
            if set(['train.0.txt', 'valid.0.txt']).issubset(set(filenames)):
                train_valid_cv = [os.path.join(dirpath, 'train.0.txt'), os.path.join(dirpath, 'valid.0.txt')]
                test_cv = os.path.join(dirpath, 'test.0.txt')

            elif 'pos' in dirpath:
                movie_root_path = dirpath.split('pos')[0]


    #Process train and valid path
    for p in train_valid_cv:
        with open(p, 'rb') as f:
            for line in f:
                train_val_file_list.append(line.decode('utf8').strip('\n').split('  '))

    #Process test
    with open(test_cv, 'rb') as f:
        for line in f:
            test_file_list.append(line.decode('utf8').strip('\n').split('  '))

    #make full path
    file_dict['train'] = [(tar, movie_root_path + p[1:]) for tar, p in train_val_file_list]
    file_dict['test'] = [(tar, movie_root_path + p[1:]) for tar, p in test_file_list]

    random.shuffle(file_dict['train'])

    for mode in ['train', 'test']:
        for tar, p in file_dict[mode]:
            with open(p, 'rb') as f:
                doc_list = []
                tar_list = []
                for line in f:
                    doc_list.append(line.decode('utf8').strip('\n'))
                    tar_list.append(int(tar))

                dat[mode]['data'].append(doc_list)
                dat[mode]['tar'].append(tar_list)


    #Convert to tfidf
    vectorizer = TfidfVectorizer()
    vectorizer.fit([item for sublist in dat['train']['data'] for item in sublist])

    for mode in dat:
        for i, doc in enumerate(dat[mode]['data']):
            dat[mode]['data'][i] = vectorizer.transform(doc)
            #concate the bias
            dat[mode]['data'][i] = np.concatenate((dat[mode]['data'][i].toarray(), 
                np.ones((dat[mode]['data'][i].shape[0], 1) ) ), axis = 1)
            dat[mode]['data'][i] = sp.sparse.csr_matrix(dat[mode]['data'][i])

    pickle.dump(dat, open(os.path.join(data_path, 'movies.pkl'), 'wb'))


def process_speech(data, data_path):
    """
    Process moview dataset
    Output:
        list of list of array with shape [number_of_documents, number_of_sentences, embedding_size]
    """

    dat = {'train': {'data':[], 'tar':[]}, 'test': {'data':[], 'tar':[]}}
    file_dict = {}
    train_val_file_list = []
    test_file_list = []

    #Use CVList, leave 
    for (dirpath, dirnames, filenames) in os.walk(data_path):
        if data.split('.')[0] in dirpath:
            if set(['stage3_training_set.txt', 'stage3_development_set.txt']).issubset(set(filenames)):
                train_valid_cv = [os.path.join(dirpath, 'stage3_training_set.txt'), 
                    os.path.join(dirpath, 'stage3_development_set.txt')]
                test_cv = os.path.join(dirpath, 'stage3_test_set.txt')

            elif 'training_set' in dirpath and 'data_stage_three_sent_SPEAKER' not in dirpath:
                speech_root_path = dirpath.split('training_set')[0]

    #Process train and valid path
    for p in train_valid_cv:
        tmp = []
        with open(p, 'rb') as f:
            if 'stage3_training_set.txt' in p:
                prefix = 'training_set/'
            else:
                prefix = 'development_set/'

            for line in f:
                tmp.append(line.decode('utf8').strip('\n').split('  '))

        tmp = [[tar, prefix+p] for tar, p in tmp]

        train_val_file_list = train_val_file_list + tmp

    #Process test
    with open(test_cv, 'rb') as f:
        for line in f:
            test_file_list.append(line.decode('utf8').strip('\n').split('  '))

    test_file_list = [[tar, 'test_set/'+p] for tar, p in test_file_list]

    #make full path
    file_dict['train'] = [(tar, speech_root_path + p) for tar, p in train_val_file_list]
    file_dict['test'] = [(tar, speech_root_path + p) for tar, p in test_file_list]

    random.shuffle(file_dict['train'])

    for mode in ['train', 'test']:
        for tar, p in file_dict[mode]:
            with open(p, 'rb') as f:
                doc_list = []
                tar_list = []
                for line in f:
                    doc_list.append(line.decode('utf8').strip('\n'))
                    tar_list.append(int(tar))

                dat[mode]['data'].append(doc_list)
                dat[mode]['tar'].append(tar_list)


    #Convert to tfidf
    vectorizer = TfidfVectorizer()
    vectorizer.fit([item for sublist in dat['train']['data'] for item in sublist])

    for mode in dat:
        for i, doc in enumerate(dat[mode]['data']):
            dat[mode]['data'][i] = vectorizer.transform(doc)
            #concate the bias
            dat[mode]['data'][i] = np.concatenate((dat[mode]['data'][i].toarray(), 
                np.ones((dat[mode]['data'][i].shape[0], 1) ) ), axis = 1)
            dat[mode]['data'][i] = sp.sparse.csr_matrix(dat[mode]['data'][i])

    pickle.dump(dat, open(os.path.join(data_path, 'speech.pkl'), 'wb'))




def prepare_data(data_path):

    """Major fn to process data"""

    if not os.path.isdir(data_path):
        os.mkdir(data_path)

    #The expected dataset name
    data_name = ['processed_stars.tar.gz', 'sle_sent_movieReviews.tar.gz', 'sle_sent_convote.tar.gz']

    for data in data_name:
        if os.path.isfile(os.path.join(data_path, data)):
            pass
        else:
            download_data(data, data_path)
            if data == 'processed_stars.tar.gz':
                process_stars()
            elif data == 'sle_sent_movieReviews.tar.gz':
                process_movies(data, data_path)
            else:
                process_speech(data, data_path)


def main():
    prepare_data('data/')

if __name__ == '__main__':
    main()