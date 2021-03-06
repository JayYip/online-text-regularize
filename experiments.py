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

def OMSA_CV(train_dat, test_dat, regularizer, delta, eta, regularize_type, loss):
    """Wrap up fn for OMSA training"""
    score_dict = collections.defaultdict(list)

    for l in loss:

        for algo in [1,2,3]:

            kf = KFold(n_splits=5, shuffle=True)

            #The X and y is sentence level, need to convert to doc level when 
            #   calculating the score.
            X, y = train_dat
            test_X, sen_test_y = test_dat
            test_y = sen2doc(sen_test_y)

            start = time.time()
            for train_index, test_index in kf.split(X):

                #Init the model
                model = OMSA(regularizer, delta, eta, regularize_type, loss = l, algo = algo)

                #Set epoch
                model.fit(X[train_index], y[train_index], epoch = 3, decay_factor = 2)
                score_dict['Algo %d ' % algo + l + ' CV_Score'].append(
                    metrics.accuracy_score(sen2doc(y[test_index]), model.predict(X[test_index])))

                score_dict['Algo %d ' % algo + l + ' Test_Score'].append(
                    metrics.accuracy_score(test_y, model.predict(test_X)))

            end = time.time()

            #Mean
            score_dict['Algo %d ' % algo + l + ' CV_Score'] = np.mean(
                score_dict['Algo %d ' % algo + l + ' CV_Score'])
            score_dict['Algo %d ' % algo + l + ' Test_Score'] = np.mean(
                score_dict['Algo %d ' % algo + l + ' Test_Score'])
            score_dict['Algo %d ' % algo + l + ' time'] = end - start

    return score_dict

def baseline_CV(train_dat, test_dat, regularizer, regularize_type, eta, loss):

    #Create parameters grid
    #parameters = {'regularizer': regularizer}
    #parameters_grid = ParameterGrid(parameters)

    def get_score(model):

        score_dict = collections.defaultdict(list)

        for l in loss:

            final_cv_score = []
            final_test_score = []

            start = time.time()

            #Loop through parm gird
            for i, reg in enumerate(regularizer):

                cv_score = []
                test_score = []

                kf = KFold(n_splits=5, shuffle=True)

                #The X and y is sentence level, need to convert to doc level when 
                #   calculating the score.
                X, y = train_dat
                test_X, sen_test_y = test_dat
                test_y = sen2doc(sen_test_y)

                model.loss = l
                model.regularizer = reg
                model.regularize_type = regularize_type[i]
                #model.set_params(**param)

                #5 Fold CV
                for train_index, test_index in kf.split(X):

                    #Init the model parm
                    model.trained = False

                    model.fit(X[train_index], y[train_index], epoch = 1)
                    cv_score.append(
                        metrics.accuracy_score(sen2doc(y[test_index]), model.predict(X[test_index])))

                    test_score.append(
                        metrics.accuracy_score(test_y, model.predict(test_X)))

                final_cv_score.append(np.mean(cv_score))
                final_test_score.append(np.mean(test_score))

            end = time.time()

            #Get the best estimator
            max_ind = np.argmax(final_cv_score)
            score_dict[l + ' CV_Score'] = final_cv_score[max_ind]
            score_dict[l + ' Test_Score'] = final_test_score[max_ind]
            score_dict[l + ' time'] = end - start

        return score_dict

    adam_model = RegAdam(0.1, eta, 'w')
    adam_score_dict = get_score(adam_model)

    ada_model = RegAdaGrad(0.1, eta, 'w')
    adagrad_score_dict = get_score(ada_model)

    return adagrad_score_dict, adam_score_dict


def news_experiments(regularizer, delta, eta, regularize_type, loss):
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

    result = collections.defaultdict(dict)

    for cat_name in cat_name_list:

        #OMSA
        train_dat = new_stream.get_dat(cat_name, 'train')
        test_dat = new_stream.get_dat(cat_name, 'test')
        omsa_score_dict = OMSA_CV(train_dat, test_dat, regularizer, delta, eta, regularize_type, loss)
        #baseline
        adagrad_score_dict, adam_score_dict = baseline_CV(train_dat, test_dat, regularizer, regularize_type, eta, loss)

        result[' vs '.join(cat_name)]['OMSA'] = omsa_score_dict
        result[' vs '.join(cat_name)]['Adagrad'] = adagrad_score_dict
        result[' vs '.join(cat_name)]['Adam'] = adam_score_dict

    return result

def movie_experiments(regularizer, delta, eta, regularize_type, loss):
    result = dict()
    data_path = os.path.join('data', 'movies.pkl')

    #Set up class
    new_stream = StreamMovies(data_path)

    result = collections.defaultdict(dict)

    train_dat = new_stream.get_dat('train')
    test_dat = new_stream.get_dat('test')
    omsa_score_dict = OMSA_CV(train_dat, test_dat, regularizer, delta, eta, regularize_type, loss)
    #baseline
    adagrad_score_dict, adam_score_dict = baseline_CV(train_dat, test_dat, regularizer, regularize_type, eta, loss)

    result['Movies Sentiment']['OMSA'] = omsa_score_dict
    result['Movies Sentiment']['Adagrad'] = adagrad_score_dict
    result['Movies Sentiment']['Adam'] = adam_score_dict

    return result

def speech_experiments(regularizer, delta, eta, regularize_type, loss):
    result = dict()
    data_path = os.path.join('data', 'speech.pkl')

    #Set up class
    new_stream = StreamSpeech(data_path)

    result = collections.defaultdict(dict)

    train_dat = new_stream.get_dat('train')
    test_dat = new_stream.get_dat('test')
    omsa_score_dict = OMSA_CV(train_dat, test_dat, regularizer, delta, eta, regularize_type, loss)
    #baseline
    adagrad_score_dict, adam_score_dict = baseline_CV(train_dat, test_dat, regularizer, regularize_type, eta, loss)

    result['Speech Sentiment']['OMSA'] = omsa_score_dict
    result['Speech Sentiment']['Adagrad'] = adagrad_score_dict
    result['Speech Sentiment']['Adam'] = adam_score_dict

    return result


def print_score(score_dict):
    for w in score_dict:
        print('########################################################')
        print(w + ' :')
        for s in score_dict[w]:
            print('    ' + s + ' : ')
            for l in score_dict[w][s]:
                print('        ' + l + ' : ', score_dict[w][s][l])
        print('########################################################')


def main():
    
    regularizer = np.exp2(range(-6, 7, 1))
    delta = 0.05
    eta = 0.01
    regularize_type = ['sq'] * regularizer.shape[0]
    loss = ['logit', 'square', 'hinge']

    #experiments fn returns a score dict
    score_dict = news_experiments(regularizer, delta, eta, regularize_type, loss)
    with io.open('20News.json', 'w', encoding='utf8') as fp:
        json.dump(score_dict, fp, ensure_ascii=False)
    print_score(score_dict)


    score_dict = movie_experiments(regularizer, delta, eta, regularize_type, loss)
    with io.open('MovieReview.json', 'w', encoding='utf8') as fp:
        json.dump(score_dict, fp, ensure_ascii=False)
    print_score(score_dict)

    score_dict = speech_experiments(regularizer, delta, eta, regularize_type, loss)
    with io.open('Speech.json', 'w', encoding='utf8') as fp:
        json.dump(score_dict, fp, ensure_ascii=False)
    print_score(score_dict)

if __name__ == '__main__':
    main()

