from numpy import *

class kNN():

    def __init__(self):
        self.range = 0 # using in feature scaling
        self.min = 0 # using in feature scaling
        self.original_X  = 0 # using to record the original X
        self.fsed_X = 0 # using to record the data after featureScaling
        self.labels = 0 # using to record the labels
        print('Initializ a kNN module.')

    def fit(self, X, y, isScaled=False):
        '''Input X, y. X are numerical features and y are labels.'''
        if X.shape[0] != y.shape[0]:
            print('ERROR: The number of X and y is not match.!')
            return
        # fsed_X is X after feature scaling
        self.original_X = X
        if isScaled == False:
            self.fsed_X = self.featureScaling(X)
        else:
            self.fsed_X = X
        self.labels = y

    def featureScaling(self, X):
        '''Feature Scaling utilize (x-min)/(max-min). Return X after scaling.'''
        minValue = X.min(0)
        maxValue = X.max(0)
        self.range = maxValue - minValue
        self.min = minValue
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
        diffMatrix = self.fsed_X - tile(new_fsed_X, (self.fsed_X.shape[0], 1))
        # second, utilize Euclidian distance to compute the distance
        distance = (diffMatrix ** 2).sum(axis=1) ** 0.5
        sortedDistanceIndex = distance.argsort()
        # argsort function return the index of sorted array from samll to large
        voteList = {}
        for i in range(k):
            voteLabel = self.labels[sortedDistanceIndex[i]]
            voteList[voteLabel] = voteList.get(voteLabel, 0) + 1
        # third, sort vote list and return the largest label
        sortedVoteListKey = sorted(voteList.keys())
        return sortedVoteListKey[0]

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
        totalNumber = self.original_X.shape[0]
        correctCount = 0
        for i in range(totalNumber):
            pre = self.predict(self.original_X[i,:], k)
            if pre == self.labels[i]:
                correctCount += 1
        return correctCount / totalNumber * 100
