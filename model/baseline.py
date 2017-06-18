#Baseline model: AdaGrad and Adam

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_X_y, check_array
from sklearn.preprocessing import normalize
from sklearn import metrics
import numpy as np
import scipy as sp

from model.base import base

class RegAdaGrad(base, BaseEstimator, ClassifierMixin):
    """
    Class of AdaGrad for comparision

    Init Args:
        regularizer: numpy array of regularizer (Lambda in paper) with the shape of [vocab_size, num_regularizer]
        eta: Step size
        regularize_type: 'w': Word reg, 's': sentence reg, 'sq': square reg
        loss: 'logit' for logit loss, 'hinge' for hinge loss, 'square' for square loss
        fudge_factor: fudge factor for adagrad. 1e-8 default.

    Method: 
        self.fit: Train model with a batch of data with bias
        self.fit_single_with_bias: Train single instance with bias

    """
    def __init__(self, regularizer, eta, regularize_type = 'w', loss = 'logit', fudge_factor = 1e-8):
        self.regularizer = regularizer
        self.eta = eta
        self.regularize_type = regularize_type
        self.loss = loss
        self.fudge_factor = fudge_factor

        #Initialize model variables
        self.trained = False
        self.trained_count = 0

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
            self.w = np.zeros(shape = [X.shape[1], 1])
            self.gti=np.zeros(self.w.T.shape)


        self.trained = True

        #Get corresponding param
        regularizer = self.regularizer
        w = self.w

        #Get the gradient
        X_doc, y_doc, grad = self._get_gradient(X, y, w)

        #Adjust gradient
        self.gti+=np.square(grad)
        adjusted_grad = np.divide(grad, self.fudge_factor + np.sqrt(self.gti))

        w_half = w - (self.eta * adjusted_grad).T

        if self.regularize_type == 'w':
            w = self._word_reg_update(w_half, 1, regularizer)
        elif self.regularize_type == 's':
            w = self._sentence_reg_update(X, w, w_half, 1, regularizer)
        elif self.regularize_type == 'sq':
            w = self._square_reg_update(w_half[:, :-1], 1, regularizer)

        self.w = w

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
            self.w = np.zeros(shape = [X.shape[1], 1])
            self.gti=np.zeros(self.w.T.shape)


        self.trained = True

        #Get corresponding param
        regularizer = self.regularizer
        w = self.w

        #Get the gradient
        X_doc, y_doc, grad = self._get_gradient(X, y, w)

        #Adjust gradient
        self.gti+=np.square(grad)
        adjusted_grad = np.divide(grad, self.fudge_factor + np.sqrt(self.gti))

        w_half = w - (self.eta * adjusted_grad).T

        #Reg exclude the bias term
        if self.regularize_type == 'w':
            w = self._word_reg_update(w_half[:, :-1], 1, regularizer)
        elif self.regularize_type == 's':
            w = self._sentence_reg_update(X, w[:, :-1], w_half[:, :-1], 1, regularizer)
        elif self.regularize_type == 'sq':
            w = self._square_reg_update(w_half[:, :-1], 1, regularizer)

        w = np.concatenate((w, w_half[:, -1]), axis = 1)

        self.w = w

    def predict_single(self, X):

        X_doc = sp.sparse.csr_matrix.sum(X, 0)
        y_predict = int(np.sign(np.dot(X_doc, self.w)))

        if y_predict == 0:
            y_predict = 1

        return y_predict

    def predict_single_with_bias(self, X):

        X_doc = sp.sparse.csr_matrix.sum(X, 0)
        y_predict = int(np.sign(np.dot(X_doc, self.w)))

        if y_predict == 0:
            y_predict = 1

        return y_predict

class RegAdam(base, BaseEstimator, ClassifierMixin):
    """    
    Class of Adam for comparision

    Init Args:
        regularizer: numpy array of regularizer (Lambda in paper) with the shape of [vocab_size, num_regularizer]
        eta: Step size
        regularize_type: 'w': Word reg, 's': sentence reg, 'sq': square reg
        loss: 'logit' for logit loss, 'hinge' for hinge loss, 'square' for square loss
        beta1: beta1 in Adam. 0.9 default.
        beta2: beta2 in Adam. 0.999 default.
        epsilon: epsilon in Adam. 1e-8 default.
        batch_size: used to compute the number of iteration. 1 default

    Method: 
        self.fit: Train model with a batch of data with bias
        self.fit_single_with_bias: Train single instance with bias
        """
    def __init__(self, regularizer, eta, regularize_type = 'w', 
        loss = 'logit', beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, batch_size = 1):

        self.regularizer = regularizer
        self.eta = eta
        self.regularize_type = regularize_type
        self.loss = loss
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.batch_size = batch_size

        #Initialize model variables
        self.trained = False
        self.trained_count = 0

        assert eta > 0, 'eta should be bigger than 0, %s found.' % str(eta)

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
            self.w = np.zeros(shape = [X.shape[1], 1])
            self.M = np.zeros_like(self.w)
            self.R = np.zeros_like(self.w)


        self.trained = True

        #Get corresponding param
        regularizer = self.regularizer
        w = self.w

        #Sum X to doc lvl
        X_doc, y_doc, grad = self._get_gradient(X, y, w)

        #Adjust gradient
        self.M = self.beta1 * self.M + (1. - self.beta1) * grad.T
        self.R = self.beta2 * self.R + (1. - self.beta2) * np.square(grad.T)


        m_k_hat = self.M / (1. - self.beta1**(int(self.trained_count / self.batch_size)))
        r_k_hat = self.R / (1. - self.beta2**(int(self.trained_count / self.batch_size)))

        adjusted_grad = np.divide(m_k_hat, (np.sqrt(r_k_hat) + self.epsilon)).T

        w_half = w - (self.eta * adjusted_grad).T

        #Reg exclude the bias term
        if self.regularize_type == 'w':
            w = self._word_reg_update(w_half[:, :-1], 1, regularizer)
        elif self.regularize_type == 's':
            w = self._sentence_reg_update(X, w[:, :-1], w_half[:, :-1], 1, regularizer)
        elif self.regularize_type == 'sq':
            w = self._square_reg_update(w_half[:, :-1], 1, regularizer)

        w = np.concatenate((w, w_half[:, -1]), axis = 1)

        self.w = w

    def predict_single(self, X):

        X_doc = sp.sparse.csr_matrix.sum(X, 0)
        y_predict = int(np.sign(np.dot(X_doc, self.w)))

        if y_predict == 0:
            y_predict = 1

        return y_predict

    def predict_single_with_bias(self, X):

        X_doc = sp.sparse.csr_matrix.sum(X, 0)
        y_predict = int(np.sign(np.dot(X_doc, self.w)))

        if y_predict == 0:
            y_predict = 1

        return y_predict