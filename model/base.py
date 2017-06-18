# Base class for models

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_X_y, check_array
from sklearn.preprocessing import normalize
from sklearn import metrics
import numpy as np
import scipy as sp

class base():
    """docstring for base"""
    def __init__(self):
        pass

    def _loss(self, y, y_predict):
        """Calculate the empirical loss"""
        if self.loss == 'logit':
            return np.log(1+np.exp(- y * y_predict))
        elif self.loss == 'hinge':
            return np.max([0, 1 - y*y_predict])
        elif self.loss == 'square':
            return np.square(y - y_predict)

    def _get_gradient(self, X, y, w):

        #Sum X to doc lvl
        X_doc = sp.sparse.csr_matrix.sum(X, 0)
        y_doc = y[0]

        #Prediction
        y_predict = int(np.sign(np.dot(X_doc, w)))

        if y_predict == 0:
            y_predict = 1

        #Calculate the grad
        if self.loss == 'logit':
            grad = -y_doc * X_doc / (1 + np.exp(y_doc * y_predict))
        elif self.loss == 'hinge':
            grad = -y_doc*X_doc
            grad[grad < 0] = 0
        elif self.loss == 'square':
            grad = (y_predict - y_doc) * X_doc

        return (X_doc, y_doc, grad)

    def _word_reg_update(self, w_half, p, regularizer):

        positive_part = np.abs(w_half) - (self.eta / p) * regularizer
        positive_part[positive_part < 0] = 0
        return np.multiply(np.sign(w_half), positive_part)

    def _square_reg_update(self, w_half, p, regularizer):
        return w_half / (1 + self.eta * regularizer / p) 

    def _sentence_reg_update(self, X, w, w_half, p, regularizer):
        
        #TODO
        def _sen_update_fn(row, w_half, p, regularizer):
            ind = row != 0

            w_half_sen = w_half[ind]
            positive_part = 1 - self.eta * regularizer / (p * np.norm(w_half_sen))
            positive_part[positive_part < 0] = 0
            w_sen = w_half_sen * positive_part

            w[ind] = w_sen

    def fit_dep(self, X1, y1):
        for i, row in enumerate(X1):
            self.fit_single(row, y1[i])

    def fit(self, X1, y1, epoch = 1, decay_factor = None):
        for i in range(epoch):
            permute = np.random.permutation(len(X1))
            
            for j in permute:
                self.fit_single_with_bias(X1[j], y1[j])

            #Add step size decay every epoch
            if decay_factor:
                self.eta = self.eta / decay_factor

    def predict_dep(self, X):
        p = []
        for row in X:
            p.append(self.predict_single(row))
        return p   

    def predict(self, X):
        p = []
        for row in X:
            p.append(self.predict_single_with_bias(row))
        return p  
