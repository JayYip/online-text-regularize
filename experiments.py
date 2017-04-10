# Train the model

from model.online_text_regularize import OnlineTextReg
from preprocess import Stream20News

import numpy as np


def train_20news(train_gen, regularizer, delta, eta, regularize_type):
    """Wrap up fn for 20news training"""


    model = OnlineTextReg(regularizer, delta, eta, regularize_type)

    X = 0
    while True:
        X, y = next(train_gen, (None, None))

        if X is None:
            break

        model.fit(X = X, y = y)

    return model
    

def main():
    
    data_path = 'data/'

    #Set up class
    new_stream = Stream20News(data_path)

    #Set up generator
    cat_name = ['comp.sys.mac.hardware', 'comp.sys.ibm.pc.hardware']
    regularizer = np.array([0.1, 0.2, 0.3])
    delta = 0.5
    eta = 0.1
    regularize_type = ['w', 'w', 'w']

    train_gen = new_stream.get_cat(cat_name, 'train')
    computer_model = train_20news(train_gen, regularizer, delta, eta, regularize_type)

    test_gen = new_stream.get_cat(cat_name, 'test')
    computer_model.eval(test_gen)

    print(computer_model.score)


if __name__ == '__main__':
    main()

