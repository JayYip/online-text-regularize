# Train the model

from model.online_text_regularize import OnlineTextReg
from data import *

import numpy as np
import os


def trainer(train_gen, regularizer, delta, eta, regularize_type):
    """Wrap up fn for 20news training"""


    model = OnlineTextReg(regularizer, delta, eta, regularize_type)

    while True:
        X, y = next(train_gen, (None, None))

        if X is None:
            break

        model.fit(X = X, y = y)

    return model
    
def news_experiments(regularizer, delta, eta, regularize_type):
    """
    Simple wrap up fn for 20news data experiment
    """
    data_path = 'data/'

    #Set up class
    new_stream = Stream20News(data_path)

    cat_name_list = [['comp.sys.mac.hardware', 'comp.sys.ibm.pc.hardware'],
                     ['rec.sport.baseball', 'rec.sport.hockey'],
                     ['sci.med', 'sci.space'],
                     ['alt.atheism', 'soc.religion.christian']]

    result = dict()

    for cat_name in cat_name_list:
        train_gen = new_stream.get_cat(cat_name, 'train')
        model = trainer(train_gen, regularizer, delta, eta, regularize_type)

        test_gen = new_stream.get_cat(cat_name, 'test')
        model.eval(test_gen)

        result['  vs  '.join(cat_name)] = model.score

    return result

def movie_experiments(regularizer, delta, eta, regularize_type):
    result = dict()
    data_path = os.path.join('data', 'movies.pkl')

    #Set up class
    new_stream = StreamMovies(data_path)

    train_gen = new_stream.get_dat('train')
    model = trainer(train_gen, regularizer, delta, eta, regularize_type)

    test_gen = new_stream.get_dat('test')
    model.eval(test_gen)
    result['Movies Sentiment'] = model.score

    return result

def speech_experiments(regularizer, delta, eta, regularize_type):
    result = dict()
    data_path = os.path.join('data', 'speech.pkl')

    #Set up class
    new_stream = StreamMovies(data_path)

    train_gen = new_stream.get_dat('train')
    model = trainer(train_gen, regularizer, delta, eta, regularize_type)

    test_gen = new_stream.get_dat('test')
    model.eval(test_gen)
    result['Speech Sentiment'] = model.score

    return result


def print_score(score_dict):
    for w in score_dict:
        print('########################################################')
        print(w + ' :')
        for s in score_dict[w]:
            print(s + ' : ', score_dict[w][s])
        print('########################################################')


def main():
    
    regularizer = np.exp2(range(-6, 6, 1))
    delta = 0.5
    eta = 0.01
    regularize_type = ['w'] * regularizer.shape[0]

    print_score(news_experiments(regularizer, delta, eta, regularize_type))
    print_score(movie_experiments(regularizer, delta, eta, regularize_type))
    print_score(speech_experiments(regularizer, delta, eta, regularize_type))

if __name__ == '__main__':
    main()

