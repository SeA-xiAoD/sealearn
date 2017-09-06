from numpy import *
from math import log
import operator
import pickle
from .TreePlotter import createPlot # coding by Peter Harrington
import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

class DecisionTree_ID3():
    '''Decision Tree based on ID3 algorithm. Do not contain pruning.'''

    def __init__(self, filename=None):
        '''You can pass a file if there is a decision tree which has been stored.'''
        self.original_X = 0 # using to record original X
        self.original_labels = 0 # using to record original labels
        self.fited_tree = 0 # using to record fited decision tree
        self.original_feature_name = 0 # using to record original features
        if filename != None:
            self.fited_tree = pickle.load(open(filename, 'rb'))
            print(self.fited_tree)
        print('Initializ a ID3 decision tree module.')

    def __calInfoEnt(self, y):
        '''Function to calculate the infomation entropy.'''
        num = len(y)
        label_count = {}
        for current_label in y:
            if current_label not in label_count.keys():
                label_count[current_label] = 0
            label_count[current_label] += 1
        info_ent = 0
        for key in label_count:
            prob = float(label_count[key])/num
            info_ent -= prob * log(prob, 2)
        return info_ent

    def __splitDataSet(self, X, y, bestFeat, value):
        '''Function to split data set by the feature which has the highest
        infomation gain.'''
        new_X = []
        new_y = []
        for i in range(0, len(X)):
            if X[i][bestFeat] == value:
                new_feat_vec = X[i][:bestFeat]
                new_feat_vec.extend(X[i][bestFeat+1:])
                new_X.append(new_feat_vec)
                new_y.append(y[i])
        return new_X, new_y

    def __selectBestFeat(self, X, y):
        '''Function to select best feature to split the data set.'''
        num_feat = len(X[0])
        best_feat = -1
        best_info_gain = 0
        base_info_ent = self.__calInfoEnt(y)
        for i in range(0, num_feat):
            # to select features in  i th number
            val_list = [example[i] for example in X]
            # to get unique value of the feature using set structure
            all_posible_value = set(val_list)
            all_posible_value = list(all_posible_value)
            all_posible_value.sort()
            temp_ent = 0
            for val in  all_posible_value:
                sub_X, sub_y = self.__splitDataSet(X, y, i, val)
                prob = len(sub_y) / len(y)
                temp_ent += prob * self.__calInfoEnt(sub_y)
            info_gain = base_info_ent - temp_ent
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feat = i
        return best_feat

    def __countMajorLabel(self, labelList):
        '''Function to select major label if there is no more features.'''
        count = {}
        for label in labelList:
            if label not in count.keys():
                count[label] = 0
            count[label] += 1
        sorted_count = sorted(count.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_count[0][0]

    def __createDecisionTree(self, X, y, featureName):
        '''Function to bulid ID3 decision tree.'''
        # 1. No more features
        if len(X) == 0:
            return self.__countMajorLabel(y)
        # 2. All labels are same label
        if y.count(y[0]) == len(y):
            return y[0]
        # 3. Continue to create sub decision tree
        best_feat = self.__selectBestFeat(X, y)
        best_feat_name = featureName[best_feat]
        decision_tree = {best_feat_name:{}}
        # delete used feature name
        del(featureName[best_feat])
        val_list = [example[best_feat] for example in X]
        all_posible_value = set(val_list)
        all_posible_value = list(all_posible_value)
        all_posible_value.sort()
        for val in all_posible_value:
            sub_features = featureName[:]
            sub_X, sub_y = self.__splitDataSet(X, y, best_feat, val)
            decision_tree[best_feat_name][val] = self.__createDecisionTree(sub_X, sub_y, sub_features)
        return decision_tree

    def fit(self, X, y, featureName):
        '''Input X, y. X are numerical features and y are labels.
        (X and y should be list, rather than matrix)'''
        if len(X) != len(y):
            print('ERROR: The number of X and y is not match!')
            return
        if len(X[0]) != len(featureName):
            print('ERROR: The number of X and features Name is not match!')
            return
        self.original_X = X[:]
        self.original_labels = y[:]
        self.original_feature_name = featureName[:]
        # building decision tree
        self.fited_tree = self.__createDecisionTree(X, y, featureName)
        print('The model fitting is finished!')
        return self.fited_tree

    def storeDecisionTree(self, filename):
        '''Function to store the decision tree.'''
        if filename == None:
            print('ERROR: The file name is NONE!')
            return
        if self.fited_tree == 0:
            print('ERROR: The model is not fited!')
            return
        fw = open(filename, 'wb')
        pickle.dump(self.fited_tree, fw)
        fw.close()

    def predict(self, features, decisionTree=None):
        '''Function to predict the label of given features.'''
        if self.fited_tree == 0:
            print('ERROR: The model is not fited!')
            return
        if len(features) != len(self.original_X[0]):
            print("ERROR: The number of input features is not equal to data's!")
            return
        if decisionTree == None:
            decisionTree = self.fited_tree
        first_key = list(decisionTree.keys())[0]
        second_dict = decisionTree[first_key]
        # to get the position of first_key in feature name
        feature_index = self.original_feature_name.index(first_key)
        for key in second_dict.keys():
            if features[feature_index] == key:
                if type(second_dict[key]).__name__ == 'dict':
                    class_label = self.predict(features, second_dict[key])
                else:
                    class_label = second_dict[key]
        return class_label

    def precision(self):
        '''Function to use original data to predict labels and compare to
        original labels to calculate the precision * 100% of the tree.'''
        if self.fited_tree == 0:
            print('ERROR: The model is not fited!')
            return
        correct_count = 0
        for i in range(0, len(self.original_X)):
            pre_label = self.predict(self.original_X[i])
            if pre_label == self.original_labels[i]:
                correct_count += 1
        return correct_count / len(self.original_X) * 100

    def drawDecisionTree(self):
        '''Function to draw the decision tree which is already fited using matplotlib.'''
        createPlot(self.fited_tree)
