from numpy import *

class AdaBoost():
    '''A simple AdaBoost algorithm using decision stump to be the weak learner.'''

    def __init__(self):
        self.original_X = 0
        self.original_labels = 0
        self.classifier_list = 0
        print('Initializ a AdaBoost module using decision stump algorithm.')

    def __stumpClassify(self, X, dimention, threshold, relation):
        '''Function to classify X using the decision stump.'''
        r_array = zeros((shape(X)[0], 1))
        if relation == '>':
            r_array[X[:, dimention] <= threshold] = -1.0
            r_array[X[:, dimention] > threshold] = 1.0
        else:
            r_array[X[:, dimention] > threshold] = -1.0
            r_array[X[:, dimention] <= threshold] = 1.0
        return r_array

    def __buildStump(self, X, y, D, steps=10):
        '''Function to select the lowest weighted error to build a decision stump.'''
        X = mat(X)
        y = mat(y).T
        m, n = shape(X)
        best_stump = {}
        best_prediction = mat(zeros((m, 1)))
        min_error = inf
        for i in range(0, n):
            column_min = X[:, i].min()
            column_max = X[:, i].max()
            step_size = (column_max - column_min)/steps
            for j in range(0, steps + 1):
                for relation in ['>', '<']:
                    threshold = column_min + j * step_size
                    pre_vals = self.__stumpClassify(X, i, threshold, relation)
                    # if predict is right, put error array to 0
                    error_array = mat(ones((m, 1)))
                    error_array[pre_vals == y] = 0
                    weighted_error = D.T * error_array
                    if weighted_error < min_error:
                        min_error = weighted_error
                        best_prediction = pre_vals
                        best_stump['dimention'] = i
                        best_stump['threshold'] = threshold
                        best_stump['relation'] = relation
        return best_stump, min_error, best_prediction

    def __buildAdaBoost(self, X, y, iteration=5):
        '''Function to build AdaBoost algorithm using decision stump.'''
        classifier_list = []
        m = shape(X)[0]
        D = mat(ones((m, 1))/m)
        classification_estimation = mat(zeros((m, 1)))
        for i in range(0, iteration):
            best_stump, error, class_estimation = self.__buildStump(X, y, D)
            alpha = float(0.5 * log((1.0 - error)/max(error, 1e-16)))
            best_stump['alpha'] = alpha
            classifier_list.append(best_stump)
            expon = multiply(mat(y).T * alpha * -1, class_estimation)
            D = multiply(D, exp(expon))
            D = D/D.sum()
            classification_estimation += class_estimation * alpha
            error = multiply(sign(classification_estimation) != mat(y).T, ones((m, 1)))
            error_rate = error.sum()/m
            if error_rate == 0.0:
                break
        return classifier_list

    def fit(self, X, y):
        '''Input X and y.'''
        if len(X) != len(y):
            print('ERROR: The number of X and y is not match!')
            return
        self.original_X = X
        self.original_labels = y
        D = ones((len(X), 1))/len(X)
        self.classifier_list = self.__buildAdaBoost(X, y)
        print(self.classifier_list)
        print('The model fitting is finished!')

    def __oneStumpClassify(self, new_X, dimention, threshold, relation):
        '''Function using to classify one sample from input.'''
        r_estimation = 0
        if relation =='>':
            if new_X[dimention] > threshold:
                r_estimation = 1.0
            else:
                r_estimation = -1.0
        else:
            if new_X[dimention] <= threshold:
                r_estimation = 1.0
            else:
                r_estimation = -1.0
        return r_estimation

    def predict(self, new_X):
        '''Function to predict the label based on new X.'''
        if self.classifier_list == 0:
            print('ERROR: The model is not fited!')
            return
        class_estimation = 0.0
        for i in range(len(self.classifier_list)):
            predict_estimation = self.__oneStumpClassify(new_X, self.classifier_list[i]['dimention'],
            self.classifier_list[i]['threshold'], self.classifier_list[i]['relation'])
            class_estimation += float(predict_estimation) * self.classifier_list[i]['alpha']
        return sign(class_estimation)

    def correctRate(self):
        '''Function to predict labels of original input X, then using new labels
        to compare with original labels, and output correct rate of this model.'''
        if self.classifier_list == 0:
            print('ERROR: The model is not fited!')
            return
        correct_count = 0
        for i in range(0, len(self.original_X)):
            label = self.predict(self.original_X[i])
            if label == self.original_labels[i]:
                correct_count += 1
        return correct_count / len(self.original_X) * 100
