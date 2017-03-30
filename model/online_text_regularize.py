from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_X_y, check_array
from sklearn.preprocessing import normalize
import numpy as np

class OnlineTextReg(BaseEstimator, ClassifierMixin):
    """
    Online Model Selection Algorithm for Spare Regularizer
    
    Args:
        regularizer: numpy array of regularizer (Lambda in paper)
        delta: Smoothing parameter. [0,1]
        eta: Step size
    """
    def __init__(self, regularizer, delta, eta, regularize_type = 'word'):
        super(OnlineTextReg, self).__init__()
        self.regularizer = regularizer
        self.delta = delta
        self.eta = eta
        self.regularize_type = regularize_type

        #Initialize model variables
        self.w = None
        self.omega = None
        self.p = None
        self.q = None

        assert isinstance(regularizer, np.ndarray), 'Regularizer should be a numpy ndarray, %s found.' % type(regularizer)

        assert delta >= 0 and delta <= 1, 'delta should be between 0 to 1, %s found.' % str(delta)

        assert eta > 0, 'eta should be bigger than 0, %s found.' % str(eta)

        assert regularize_type in ('word', 'sentence'), 'regularize_type should be "word" or "sentence", %s found.' % regularize_type

        self.dimension_size = self.regularizer.shape[0]

    def _loss(self, y, y_predict):
        """Calculate the empirical loss"""
        return np.log(1+np.exp(- y * y_predict))

    def fit(self, X, y):

        #Check X and y have the valid shape
        X, y = check_X_y(X, y)

        assert isinstance(X, np.ndarray), 'Regularizer should be a numpy ndarray, %s found.' % type(X)
        assert isinstance(y, np.ndarray), 'Regularizer should be a numpy ndarray, %s found.' % type(y)

        assert X.shape[1] == self.dimension_size, 'The dimension of X and regularizer not aligned. (:,%d) expected, (:,%d) found.' % (self.dimension_size, X.shape[1])

        #Initialize variables
        if not self.w:
            self.w = np.zeros(self.regularizer)
            self.omega = np.ones(self.regularizer)
            self.p = np.ones(self.regularizer) / self.dimension_size
            self.q = np.ones(self.regularizer) / self.dimension_size

        #Sampling i and j
        i = np.random.choice(np.arange(self.dimension_size), 1, self.q)
        j = np.random.choice(np.arange(self.dimension_size), 1, self.q)

        #Prediction
        y_predict = np.dot(X, self.w)

        #Calculate the grad
        w_half = self.w - (self.eta / self.p) * (y_predict - y) 

        if self.regularize_type == 'word':
            positive_part = np.abs(w_half) - (self.eta / self.p) * self.regularizer
            positive_part[positive_part < 0] = 0
            self.w = np.sign(w_half) * positive_part

        loss = self._loss(y, y_predict)
        norm = np.linalg.norm(self.w, 1)

        self.omega = self.omega * np.exp(-self.eta * (loss + self.regularizer *  norm) / self.p)
        self.q = normalize(self.omega)

        self.p = (1 - self.delta) * self.q + self.delta / self.dimension_size



    def predict(self, X):

        #Input validation
        X = check_array(X)

        return np.dot(X, self.w)

        
def main():
    check_estimator(OnlineTextReg)

if __name__ == '__main__':
    main()