# The Online Model Selection Algorithm


from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_X_y, check_array
from sklearn.preprocessing import normalize
from sklearn import metrics
import numpy as np
import scipy as sp

class OnlineTextReg(BaseEstimator, ClassifierMixin):
    """
    Online Model Selection Algorithm for Spare Regularizer

    Init Args:
        regularizer: numpy array of regularizer (Lambda in paper) with the shape of [vocab_size, num_regularizer]
        delta: Smoothing parameter. [0,1]
        eta: Step size
    """
    def __init__(self, regularizer, delta, eta, regularize_type, loss = 'logit'):
        super(OnlineTextReg, self).__init__()
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

    def _sentence_reg_update(self, X, w, w_half, p, regularizer):
        
        #TODO
        def _sen_update_fn(row, w_half, p, regularizer):
            ind = row != 0

            w_half_sen = w_half[ind]
            positive_part = 1 - self.eta * regularizer / (p * np.norm(w_half_sen))
            positive_part[positive_part < 0] = 0
            w_sen = w_half_sen * positive_part

            w[ind] = w_sen


            


    def fit(self, X, y):
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
        i = np.random.choice(np.arange(self.regularizer_size), 1, p = self.p)
        j = np.random.choice(np.arange(self.regularizer_size), 1, p = self.q)


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
            grad = np.max([0, -y_doc*y_predict])
        elif self.loss == 'square':
            grad = (y_predict - y_doc) * X_doc

        w_half = w - (self.eta / p) * grad * np.transpose(X_doc != 0)

        if self.regularize_type[i] == 'w':
            w = self._word_reg_update(w_half, p, regularizer)
        elif self.regularize_type[i] == 's':
            w = self._sentence_reg_update(X, w, w_half, p, regularizer)

        self.w[:, i] = w

        loss = self._loss(y_doc, y_predict)
        norm = np.linalg.norm(self.w, 1)

        omega = omega * np.exp(-self.eta * (loss + regularizer *  norm) / p)

        self.omega[i] = omega
        self.q = self.omega / self.omega.sum()

        p = (1 - self.delta) * q + self.delta / self.regularizer_size
        self.p[i] = p
        self.p = self.p / self.p.sum()

    #def fit(self, X, y):
    #    for i, row in enumerate(X):
    #        self._fit(row, y[i])


    def predict(self, X):

        #Sampling i and j
        i = np.random.choice(np.arange(self.regularizer_size), 1, p = self.p)
        j = np.random.choice(np.arange(self.regularizer_size), 1, p = self.q)
        w = self.w[:, i]


        X_doc = sp.sparse.csr_matrix.sum(X, 0)
        y_predict = int(np.sign(np.dot(X_doc, w)))

        if y_predict == 0:
            y_predict = 1

        return y_predict

    #def predict(self, X):
    #    for i, row in enumerate(X):
    #        self._fit(row, y[i])

    def eval(self, gen):

        """
        This function is used to evaluate the model. This function will
        create a score attribute of the model.
        Args:
            gen: A generator that yields X, y.
        """

        y_predict = []
        y_true = []

        while True:
            X, y = next(gen, (None, None))

            if X is None:
                break
            y_true.append(y[0])
            y_predict.append(self.predict(X))

        self.score = {'accuracy_score': metrics.accuracy_score(y_true, y_predict),
                            'precision_score': metrics.precision_score(y_true, y_predict),
                            'recall_score': metrics.recall_score(y_true, y_predict),
                            'f1_score': metrics.f1_score(y_true, y_predict)}

         

    def _shape_check(self):
        """Test the shapes of parameters"""

        m = self.regularizer.shape[1]
        v = self.regularizer.shape[0]

        assert self.w.shape == self.regularizer.shape
        assert self.omega.shape[0] == m
        assert self.p.shape[0] == m
        assert self.q.shape[0] == m


        
def main():
    check_estimator(OnlineTextReg)

if __name__ == '__main__':
    main()

