from numpy import *
import matplotlib.pyplot as plt

class LogisticRegression():
    '''A simple logistic regression that contains a linear function to
    fit the model and using in binary classification. Due to the model
    is too simple to fit complex data, the precision is not satisfying.'''

    def __init__(self):
        self.original_X = 0
        self.original_labels = 0
        self.weights = 0
        print('Initializ a Logistic Regression module.')

    def __sigmoid(self, z):
        '''Function to return the value based on sigmoid function.'''
        return 1 / (1 + exp(-z))

    def __gradientAscent(self, X, y, cycles):
        '''Function to do gradient acsent.'''
        X = mat(X)
        temp_one = ones((shape(X)[0], 1))
        X = column_stack((temp_one, X))
        y = mat(y)
        y = y.transpose()
        alpha = 0.001
        weights = ones((shape(X)[1], 1))
        for i in range(0, cycles):
            h = self.__sigmoid(X * weights)
            error = y - h
            weights = weights + alpha * X.transpose() * error
        return weights

    def fit(self, X, y, cycles=500):
        '''Input features X and labels y, then the model will fit a suitable
        line to divide the data into two pieces.  The default cycles the algorithm
        will do is 500 and you can adjust it if you think it is necessary.'''
        if len(X) != len(y):
            print('ERROR: The number of X and y is not match!')
            return
        self.original_X = X
        self.original_labels = y
        self.weights = self.__gradientAscent(X, y, cycles)
        print('The model fitting is finished!')

    def predict(self, new_X):
        '''Function to predict the label based on new input X.'''
        if self.weights.all() == 0:
            print('ERROR: The model is not fited!')
            return
        new_X = list(new_X)
        new_X.insert(0, 1)
        new_X = mat(new_X)
        return 1 if self.__sigmoid(new_X * self.weights) >= 0.5 else 0

    def precision(self):
        '''Function to predict labels of original input X, then using new labels
        to compare with original labels, and output precision of this model'''
        if self.weights.all() == 0:
            print('ERROR: The model is not fited!')
            return
        correct_count = 0
        for i in range(0, len(self.original_X)):
            if self.predict(self.original_X[i]) == self.original_labels[i]:
                correct_count += 1
        return correct_count / len(self.original_X) * 100
