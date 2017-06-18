# The Online Model Selection Algorithm


from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_X_y, check_array
from sklearn.preprocessing import normalize
from sklearn import metrics
import numpy as np
import scipy as sp

from model.base import base

class OMSA(base, BaseEstimator, ClassifierMixin):
    """
    Online Model Selection Algorithm for Spare Regularizer

    Init Args:
        regularizer: numpy array of regularizer (Lambda in paper) with the shape of [vocab_size, num_regularizer]
        delta: Smoothing parameter. [0,1]
        eta: Step size
        regularize_type: 'w': Word reg, 's': sentence reg, 'sq': square reg
        loss: 'logit' for logit loss, 'hinge' for hinge loss, 'square' for square loss
        decay_factor: Step size decay factor. None default.

    Method: 
        self.fit: Train model with a batch of data with bias
        self.fit_single_with_bias: Train single instance with bias
    """
    def __init__(self, regularizer, 
        delta, eta, regularize_type, 
        loss = 'logit', algo = 1):
        self.regularizer = regularizer
        self.delta = delta
        self.eta = eta
        self.regularize_type = regularize_type
        self.loss = loss
        self.algo = algo

        self.regularizer_size = self.regularizer.shape[0]

        #Initialize model variables
        self.trained = False
        self.trained_count = 0

        assert isinstance(regularizer, np.ndarray), 'Regularizer should be a numpy ndarray, %s found.' % type(regularizer)

        assert delta >= 0 and delta <= 1, 'delta should be between 0 to 1, %s found.' % str(delta)

        assert eta > 0, 'eta should be bigger than 0, %s found.' % str(eta)

    def fit_single(self, X, y):
        """
        Train the model using one instance.

        Args:
            X: numpy.ndarray with the shape of [number_of_sentence, vocab_size]. Each sentence should be in one
                row. All rows will be summed to one row when prediction.
            y: Scaler. 
        """

        self.trained_count += 1

        if not self.trained:
            self.w = np.zeros(shape = [X.shape[1], self.regularizer_size])
            self.omega = np.ones(self.regularizer_size)
            self.p = np.ones(self.regularizer_size) / self.regularizer_size
            self.q = np.ones(self.regularizer_size) / self.regularizer_size

        self.trained = True
        #Sampling i and j
        i = int(np.random.choice(np.arange(self.regularizer_size), 1, p = self.p))
        #j = int(np.random.choice(np.arange(self.regularizer_size), 1, p = self.q))


        #Get corresponding param
        regularizer = self.regularizer[i]
        w = self.w[:, i]
        p = self.p[i]
        #q = self.q[j]
        omega = self.omega[i]

        #Get the gradient
        X_doc, y_doc, grad = self._get_gradient(X, y, w)

        w_half = w - ((self.eta / p) * grad)

        if self.regularize_type[i] == 'w':
            w = self._word_reg_update(w_half, p, regularizer)
        elif self.regularize_type[i] == 's':
            w = self._sentence_reg_update(X, w, w_half, p, regularizer)


        self.w[:, i] = w

        y_predict = int(np.sign(np.dot(X_doc, w)))

        loss = self._loss(y_doc, y_predict)
        norm = np.linalg.norm(self.w, 1)

        omega = omega * np.exp(-self.eta * (loss + regularizer *  norm) / p)

        self.omega[i] = np.max([omega, 1e-8])
        self.q = self.omega / self.omega.sum()

        p = (1 - self.delta) * q + self.delta / self.regularizer_size
        self.p[i] = p
        self.p = self.p / self.p.sum()

    def fit_single_with_bias(self, X, y):
        """
        Train the model using one instance.

        Args:
            X: numpy.ndarray with the shape of [number_of_sentence, vocab_size]. Each sentence should be in one
                row. All rows will be summed to one row when prediction.
            y: Scaler. 
        """

        self.trained_count += 1

        if not self.trained:
            self.w = np.zeros(shape = [X.shape[1], self.regularizer_size])
            self.w_bar = np.zeros(shape = [X.shape[1], self.regularizer_size])
            self.omega = np.ones(self.regularizer_size)
            self.p = np.ones(self.regularizer_size) / self.regularizer_size
            self.q = np.ones(self.regularizer_size) / self.regularizer_size
            self.trained_count = 1

        self.trained = True
        #Sampling i and j
        i = int(np.random.choice(np.arange(self.regularizer_size), 1, p = self.p))


        #Get corresponding param
        regularizer = self.regularizer[i]
        w = self.w[:, i]
        w_bar = self.w_bar[:, i]
        p = self.p[i]
        omega = self.omega[i]

        #Get the gradient
        if self.algo == 2:
            X_doc, y_doc, grad = self._get_gradient(X, y, w_bar)
        else:
            X_doc, y_doc, grad = self._get_gradient(X, y, w)

        w_half = w - ((self.eta / p) * grad)

        #Reg exclude the bias term
        if self.regularize_type[i] == 'w':
            w = self._word_reg_update(w_half[:, :-1], p, regularizer)
        elif self.regularize_type[i] == 's':
            w = self._sentence_reg_update(X, w[:, :-1], w_half[:, :-1], p, regularizer)
        elif self.regularize_type[i] == 'sq':
            w = self._square_reg_update(w_half[:, :-1], p, regularizer)

        w = np.concatenate((w, w_half[:, -1]), axis = 1)

        self.w[:, i] = w

        #Algo 2
        if self.algo == 1:
            y_predict = int(np.sign(np.dot(X_doc, w.T)))
            norm = np.linalg.norm(self.w[:, i], 1)
        elif self.algo == 2:
            self.w_bar[:, i] = (self.trained_count - 1) * w_bar / self.trained_count + w / self.trained_count
            y_predict = int(np.sign(np.dot(X_doc, w.T)))
            norm = np.linalg.norm(self.w[:, i], 1)
        elif self.algo == 3:
            self.w_bar[:, i] = (self.trained_count - 1) * w_bar / self.trained_count + w / self.trained_count
            y_predict = int(np.sign(np.dot(X_doc, self.w_bar[:, i])))
            norm = np.linalg.norm(self.w_bar[:, i], 1)

        loss = self._loss(y_doc, y_predict)

        omega = omega * np.exp(-self.eta * (loss + regularizer *  norm) / p)

        #Avoid 0
        self.omega[i] = np.max([omega, 1e-12])
        self.q = self.omega / self.omega.sum()
        self.omega = self.q

        self.p = (1 - self.delta) * self.q + self.delta / self.regularizer_size
        #self.p[i] = p
        self.p = self.p / self.p.sum()


    def predict_single(self, X):

        #Sampling i and j
        i = np.random.choice(np.arange(self.regularizer_size), 1, p = self.p)
        #j = np.random.choice(np.arange(self.regularizer_size), 1, p = self.q)
        w = self.w[:, i]


        X_doc = sp.sparse.csr_matrix.sum(X, 0)
        y_predict = int(np.sign(np.dot(X_doc, w)))

        if y_predict == 0:
            y_predict = 1

        return y_predict

    def predict_single_with_bias(self, X):
        #Sampling i and j
        j = np.random.choice(np.arange(self.regularizer_size), 1, p = self.q)
        w = self.w[:, j]
        w_bar = self.w_bar[:, j]

        X_doc = sp.sparse.csr_matrix.sum(X, 0)

        if self.algo == 2:
            y_predict = int(np.sign(np.dot(X_doc, w_bar) ) )
        else:
            y_predict = int(np.sign(np.dot(X_doc, w)))

        if y_predict == 0:
            y_predict = 1

        return y_predict
        