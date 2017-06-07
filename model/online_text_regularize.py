# The Online Model Selection Algorithm


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

    def fit(self, X1, y1, epoch = 1):
        for i in range(epoch):
            print(i)
            permute = np.random.permutation(len(X1))
            #for i, row in enumerate(X1):
            #    self.fit_single_with_bias(row, y1[i])
            for j in permute:
                self.fit_single_with_bias(X1[j], y1[j])

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


class OMSA(base, BaseEstimator, ClassifierMixin):
    """
    Online Model Selection Algorithm for Spare Regularizer

    Init Args:
        regularizer: numpy array of regularizer (Lambda in paper) with the shape of [vocab_size, num_regularizer]
        delta: Smoothing parameter. [0,1]
        eta: Step size
        regularize_type: 'w': Word reg, 's': sentence reg, 'sq': square reg
    """
    def __init__(self, regularizer, delta, eta, regularize_type, loss = 'logit'):
        self.regularizer = regularizer
        self.delta = delta
        self.eta = eta
        self.regularize_type = regularize_type
        self.loss = loss

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
        j = int(np.random.choice(np.arange(self.regularizer_size), 1, p = self.q))


        #Get corresponding param
        regularizer = self.regularizer[i]
        w = self.w[:, i]
        p = self.p[i]
        q = self.q[j]
        omega = self.omega[i]

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


        w_half = w - ((self.eta / p) * grad)

        if self.regularize_type[i] == 'w':
            w = self._word_reg_update(w_half, p, regularizer)
        elif self.regularize_type[i] == 's':
            w = self._sentence_reg_update(X, w, w_half, p, regularizer)


        self.w[:, i] = w

        loss = self._loss(y_doc, y_predict)
        norm = np.linalg.norm(self.w, 1)

        omega = omega * np.exp(-self.eta * (loss + regularizer *  norm) / p)

        self.omega[i] = np.max([omega, 1e-6])
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

        #concate the bias
        #X = np.concatenate((X.toarray(), np.ones((X.shape[0], 1) ) ), axis = 1)
        #X = sp.sparse.csr_matrix(X)

        if not self.trained:
            self.w = np.zeros(shape = [X.shape[1], self.regularizer_size])
            self.omega = np.ones(self.regularizer_size)
            self.p = np.ones(self.regularizer_size) / self.regularizer_size
            self.q = np.ones(self.regularizer_size) / self.regularizer_size

        self.trained = True
        #Sampling i and j
        i = int(np.random.choice(np.arange(self.regularizer_size), 1, p = self.p))
        j = int(np.random.choice(np.arange(self.regularizer_size), 1, p = self.q))


        #Get corresponding param
        regularizer = self.regularizer[i]
        w = self.w[:, i]
        p = self.p[i]
        q = self.q[j]
        omega = self.omega[i]

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

        loss = self._loss(y_doc, y_predict)
        norm = np.linalg.norm(self.w, 1)

        omega = omega * np.exp(-self.eta * (loss + regularizer *  norm) / p)

        self.omega[i] = np.max([omega, 1e-6])
        self.q = self.omega / self.omega.sum()

        p = (1 - self.delta) * q + self.delta / self.regularizer_size
        self.p[i] = p
        self.p = self.p / self.p.sum()


    def predict_single(self, X):

        #Sampling i and j
        i = np.random.choice(np.arange(self.regularizer_size), 1, p = self.p)
        j = np.random.choice(np.arange(self.regularizer_size), 1, p = self.q)
        w = self.w[:, i]


        X_doc = sp.sparse.csr_matrix.sum(X, 0)
        y_predict = int(np.sign(np.dot(X_doc, w)))

        if y_predict == 0:
            y_predict = 1

        return y_predict

    def predict_single_with_bias(self, X):

        #concate the bias
        #X = np.concatenate((X.toarray(), np.ones((X.shape[0], 1) ) ), axis = 1)
        #X = sp.sparse.csr_matrix(X)

        #Sampling i and j
        i = np.random.choice(np.arange(self.regularizer_size), 1, p = self.p)
        j = np.random.choice(np.arange(self.regularizer_size), 1, p = self.q)
        w = self.w[:, i]


        X_doc = sp.sparse.csr_matrix.sum(X, 0)
        y_predict = int(np.sign(np.dot(X_doc, w)))

        if y_predict == 0:
            y_predict = 1

        return y_predict

         
class L1RegAdaGrad(base, BaseEstimator, ClassifierMixin):
    """docstring for L1Reg"""
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
        #concate the bias
        #X = np.concatenate((X.toarray(), np.ones((X.shape[0], 1) ) ), axis = 1)
        #X = sp.sparse.csr_matrix(X)

        if not self.trained:
            self.w = np.zeros(shape = [X.shape[1], 1])
            self.gti=np.zeros(self.w.T.shape)


        self.trained = True

        #Get corresponding param
        regularizer = self.regularizer
        w = self.w

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

        #Adjust gradient
        self.gti+=np.square(grad)
        adjusted_grad = np.divide(grad, self.fudge_factor + np.sqrt(self.gti))

        w_half = w - (self.eta * adjusted_grad).T

        #Reg exclude the bias term
        if self.regularize_type == 'w':
            w = self._word_reg_update(w_half[:, :-1], 1, regularizer)
        elif self.regularize_type == 's':
            w = self._sentence_reg_update(X, w[:, :-1], w_half[:, :-1], 1, regularizer)

        w = np.concatenate((w, w_half[:, -1]), axis = 1)

        self.w = w

    def predict_single(self, X):

        X_doc = sp.sparse.csr_matrix.sum(X, 0)
        y_predict = int(np.sign(np.dot(X_doc, self.w)))

        if y_predict == 0:
            y_predict = 1

        return y_predict

    def predict_single_with_bias(self, X):

        #concate the bias
        #X = np.concatenate((X.toarray(), np.ones((X.shape[0], 1) ) ), axis = 1)
        #X = sp.sparse.csr_matrix(X)

        X_doc = sp.sparse.csr_matrix.sum(X, 0)
        y_predict = int(np.sign(np.dot(X_doc, self.w)))

        if y_predict == 0:
            y_predict = 1

        return y_predict