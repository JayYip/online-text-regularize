# Train the model

from model.online_text_regularize import OnlineTextReg
from preprocess import Stream20News
import numpy as np

def main():
    
    data_path = 'data/'

    #Set up class
    new_stream = Stream20News(data_path)

    #Set up generator
    cat_name = ['comp.sys.mac.hardware', 'comp.sys.ibm.pc.hardware']
    computer_gen = new_stream.get_train_cat(cat_name)


    #Set up model
    regularizer = np.array([0.1, 0.2, 0.3])
    delta = 0.5
    eta = 0.1
    regularize_type = ['w', 'w', 'w']
    computer_model = OnlineTextReg(regularizer, delta, eta, regularize_type)

    for step in range(30):
        X, y = next(computer_gen)
        computer_model.fit(X = X, y = y)



if __name__ == '__main__':
    main()

