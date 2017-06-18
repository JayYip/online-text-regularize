# Train the model

from model.online_text_regularize import *
from model.baseline import *
from data import *

import numpy as np
import os
import collections
import time
import json
import io
from sklearn.model_selection import KFold, GridSearchCV, ParameterGrid
from sklearn import metrics

def sen2doc(sen):
    doc = []
    for s in sen:
        if not np.isscalar(s):
            doc.append(s[0])
        else:
            doc.append(s)
    return doc

def main():

    #Set loss and algorithm
    l = 'logit'
    algo = 1

    regularizer = np.exp2(range(-6, 7, 1))
    delta = 0.01
    eta = 0.005
    regularize_type = ['sq'] * regularizer.shape[0]
    loss = ['logit', 'square', 'hinge']

    data_path = 'data/movies.pkl'

    #Set up class
    new_stream = StreamMovies(data_path)
    cat_name = ['comp.sys.mac.hardware', 'comp.sys.ibm.pc.hardware']
    train_dat = new_stream.get_dat('train')
    test_dat = new_stream.get_dat('test')

    model = OMSA(regularizer, delta, eta, regularize_type, loss = l, algo = algo)

    X, y = train_dat
    test_X, sen_test_y = test_dat
    test_y = sen2doc(sen_test_y)

    X = X[:900]
    y = y[:900]

    model.trained = False

    #Set epoch
    for i in range(100):
        model = OMSA(regularizer, delta, eta, regularize_type, loss = l, algo = algo)
        print('Epoch %d' % i)
        model.fit(X, y, epoch = 5, decay_factor = 2)

        print('Step Size: %f' % model.eta)

        print('Train Accuracy: %f' % metrics.accuracy_score(sen2doc(y), model.predict(X)))

        print('Test Accuracy: %f' % metrics.accuracy_score(test_y, model.predict(test_X)))

        print('Probability: ', model.p)


if __name__ == '__main__':
    main()