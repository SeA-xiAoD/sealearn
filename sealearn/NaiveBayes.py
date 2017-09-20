from numpy import *

class NaiveBayes():
    '''A simple Naive Bayes classifier, using to classify text.'''

    def __init__(self):
        self.vocabulary_list = 0
        self.original_X = 0
        self.original_labels = 0
        self.original_label_list = 0
        self.probability = [] # using to record probability of each word
        self.total_words = [] # using to record the count of words of each label
        print('Initializ a Naive Bayes module.')

    def __createVocabularyList(self, dataSet):
        '''Function to create vocabulary list and get sorted list.'''
        vocabulary_list = set([])
        for document in dataSet:
            vocabulary_list |= set(document)
        sorted_vocabulary_list = list(vocabulary_list)
        sorted_vocabulary_list.sort()
        self.vocabulary_list = sorted_vocabulary_list[:]

    def __words2Vec(self, words):
        '''Function to return the vector of certain words.'''
        if self.vocabulary_list == 0:
            print('ERROR: The vocabulart list is NONE!')
            return
        r_vec = zeros(len(self.vocabulary_list))
        for word in words:
            r_vec[self.vocabulary_list.index(word)] += 1
        return r_vec

    def __words2Matrix(self, words):
        '''Function to return the matrix of certain words.'''
        r_matrix = []
        for document in words:
            r_matrix.append(self.__words2Vec(document))
        return array(r_matrix)

    def __createWordProb(self, X, y):
        '''Function to get probability of each word in tarining set.'''
        words_matrix = self.__words2Matrix(X)
        self.original_label_list = list(set(y))
        self.original_label_list.sort()
        for i in range(0, len(self.original_label_list)):
            self.probability.append(zeros(len(self.vocabulary_list)))
            self.total_words.append(0)
        self.probability = array(self.probability)
        for i in range(0, len(X)):
            for j in range(0, len(self.original_label_list)):
                if self.original_label_list[j] == y[i]:
                    self.probability[j] += words_matrix[i]
                    self.total_words[j] += sum(words_matrix[i])
        for i in range(0, len(self.original_label_list)):
            self.probability[i] /= self.total_words[i]

    def fit(self, X, y):
        '''Input X and y. X are list of words, and y are labels of each text.'''
        if len(X) != len(y):
            print('ERROR: The number of X and y is not match!')
            return
        self.original_X = X
        self.original_labels = y
        self.__createVocabularyList(X)
        self.__createWordProb(X, y)
        print('The model fitting is finished!')

    def predict(self, new_X):
        '''Function to predict label of the new list of words.'''
        if self.vocabulary_list == 0:
            print('ERROR: The model is not fited!')
            return
        new_vec = array(self.__words2Vec(new_X))
        max_prob = 0
        max_index = -1
        for i in range(0, len(self.original_label_list)):
            prob = sum(new_vec * self.probability[i])*(self.total_words[i]/sum(self.total_words))
            if prob > max_prob:
                max_prob = prob
                max_index = i
        return max_index

    def correctRate(self):
        '''Function to predict labels of original input X, then using new labels
        to compare with original labels, and output corret rate of this model'''
        if self.vocabulary_list == 0:
            print('ERROR: The model is not fited!')
            return
        correct_count = 0
        for i in range(0, len(self.original_X)):
            label = self.predict(self.original_X[i])
            if label == self.original_labels[i]:
                correct_count += 1
        return correct_count / len(self.original_X) * 100
