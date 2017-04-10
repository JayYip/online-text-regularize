# Process data for experiments
# Data Source:
# 1 http://qwone.com/~jason/20Newsgroups/ 
# 2 http://www.cs.jhu.edu/~mdredze/datasets/sentiment/ 
# 3 http://www.cs.cornell.edu/~ainur/sle-data.html

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import urllib.request
import tarfile
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import vstack
from nltk import sent_tokenize
import collections
import gensim
import numpy as np

def download_data(data, data_path):

    if data == 'processed_stars.tar.gz':
        urllib.request.urlretrieve('http://www.cs.jhu.edu/~mdredze/datasets/sentiment/processed_stars.tar.gz', 
                os.path.join(data_path, data))

    elif data == 'sle_movieReviews.tar.gz':
        urllib.request.urlretrieve('http://www.cs.cornell.edu/~ainur/data/sle_movieReviews.tar.gz', 
                os.path.join(data_path, data))


class Stream20News(object):
    """Class for streaming 20News data"""
    
    def __init__(self, data_path):
        self.data_path = data_path


    def get_cat(self, cat_name, mode):

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
            elif len(dat.data[i]) ==  1:
                dat.data[i] = dat.data[i][0]
            else:
                continue

            yield (dat.data[i], tar)


            

def process_stars():
    pass

def process_movie():
    pass


def prepare_data(data_path):

    """Major fn to process data"""

    if not os.path.isdir(data_path):
        os.mkdir(data_path)

    #The expected dataset name
    data_name = ['processed_stars.tar.gz', 'sle_movieReviews.tar.gz']

    for data in data_name:
        if os.path.isfile(os.path.join(data_path, data)):
            pass
        else:
            download_data(data, data_path)
            if data == 'processed_stars.tar.gz':
                process_stars()

            elif data == 'sle_movieReviews.tar.gz':
                process_movie()


