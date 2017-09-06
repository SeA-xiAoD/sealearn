from numpy import *
import operator

class kNN():

    def __init__(self):
        self.range = 0 # using in feature scaling
        self.min = 0 # using in feature scaling
        self.original_X  = 0 # using to record the original X
        self.fsed_X = 0 # using to record the data after __featureScaling
        self.labels = 0 # using to record the labels
        print('Initializ a kNN module.')

    def fit(self, X, y, isScaled=False):
        '''Input X, y. X are numerical features and y are labels.'''
        if X.shape[0] != y.shape[0]:
            print('ERROR: The number of X and y is not match!')
            return
        # fsed_X is X after feature scaling
        self.original_X = X
        if isScaled == False:
            self.fsed_X = self.__featureScaling(X)
        else:
            self.fsed_X = X
        self.labels = y

    def __featureScaling(self, X):
        '''Feature Scaling utilize (x-min)/(max-min). Return X after scaling.'''
        min_value = X.min(0)
        max_value = X.max(0)
        self.range = max_value - min_value
        self.min = min_value
        m = X.shape[0]
        new_X = X - tile(self.min, (m, 1))
        new_X = new_X / tile(self.range, (m, 1))
        return new_X

    def predict(self, new_X, k):
        '''Input new features and k to predict the new X's label.'''
        if len(new_X) != self.fsed_X.shape[1]:
            print("ERROR: The number of input features is not equal to data's!")
            return
        if k == 0:
            print("ERROR: The K can't be ZERO!")
            return
        if k > self.labels.shape[0]:
            print("ERROR: The K is too large!")
            return
        # first, normalize the input features
        new_fsed_X = (new_X - self.min)/self.range
        diff_matrix = self.fsed_X - tile(new_fsed_X, (self.fsed_X.shape[0], 1))
        # second, utilize Euclidian distance to compute the distance
        distance = (diff_matrix ** 2).sum(axis=1) ** 0.5
        sorted_distance_index = distance.argsort()
        # argsort function return the index of sorted array from samll to large
        voteList = {}
        for i in range(k):
            voteLabel = self.labels[sorted_distance_index[i]]
            voteList[voteLabel] = voteList.get(voteLabel, 0) + 1
        # third, sort vote list and return the largest label
        sorted_vote_list_key = sorted(voteList.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_vote_list_key[0][0]

    def precision(self, k):
        '''Utilize original data to compute the precision of this model,
        and return the precision * 100%.
        you can choose different k to observe different precision.'''
        if k > self.labels.shape[0]:
            print("ERROR: The K is too large!")
            return
        if k == 0:
            print("ERROR: The K can't be ZERO!")
            return
        total_number = self.original_X.shape[0]
        correct_count = 0
        for i in range(total_number):
            pre = self.predict(self.original_X[i,:], k)
            if pre == self.labels[i]:
                correct_count += 1
        return correct_count / total_number * 100
