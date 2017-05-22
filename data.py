#Data generator

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import vstack
from nltk import sent_tokenize
import collections
import gensim
import numpy as np
import pickle


class Stream20News(object):
    """
    Class for streaming 20News data

    dat.data: list of matrix with shape [number_of_documents, number_of_sentences, embedding_size]
    """
    
    def __init__(self, data_path):
        self.data_path = data_path


    def get_dat(self, cat_name, mode):

        dat = fetch_20newsgroups(subset = mode, categories = cat_name, 
            remove=('headers', 'footers', 'quotes'), data_home = self.data_path)

        if mode == 'train':
            self.vectorizer = TfidfVectorizer()
            self.vectorizer.fit(dat.data)

        target_list = []

        #Tokenize by sentences
        for i, d in enumerate(dat.data):
            d = d.replace('\n', ' ')
            dat.data[i] = sent_tokenize(d)

            for s, sent in enumerate(dat.data[i]):
                dat.data[i][s] = self.vectorizer.transform([sent])

            #Broadcast target to have the same shape
            tar = np.empty(len(dat.data[i]))
            if dat.target[i] == 0:
                tar.fill(-1)
            else:
                tar.fill(dat.target[i])


            if len(dat.data[i]) >  1:
                dat.data[i] = vstack(dat.data[i])
                target_list.append(tar)
            elif len(dat.data[i]) ==  1:
                dat.data[i] = dat.data[i][0]
                target_list.append(tar)
            else:
                dat.data[i] = None

        dat.data = [d for d in dat.data if d is not None]

        return np.array(dat.data), np.array(target_list)

class StreamMovies(object):
    """docstring for StreamMovies"""
    def __init__(self, data_path):
        self.data_path = data_path
        self.dat = pickle.load(open(self.data_path, 'rb'))

    def get_dat(self,mode):
        #for i, d in enumerate(self.dat[mode]['data']):
        #    yield (d, self.dat[mode]['tar'][i])
        return np.array(self.dat[mode]['data']), np.array(self.dat[mode]['tar'])

class StreamSpeech(object):
    """docstring for StreamMovies"""
    def __init__(self, data_path):
        self.data_path = data_path
        self.dat = pickle.load(open(self.data_path, 'rb'))

    def get_dat(self,mode):
        #for i, d in enumerate(self.dat[mode]['data']):
        #    yield (d, self.dat[mode]['tar'][i])
        return np.array(self.dat[mode]['data']), np.array(self.dat[mode]['tar'])
        