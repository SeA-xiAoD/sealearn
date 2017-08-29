from numpy import *

def txt2matrix(filename, split_symble='\t'):
    '''Input a file formated with .txt then convert it to a matrix.
    Your .txt can only have numerical value rather than string, so you need to
    do some preparation. (I will complete it later _(:зゝ∠)_ )
    And you can choose your own split symble which is defalut by \t.'''
    f = open(filename)
    line = f.readline() # read one line to count number of features
    line = line.strip()
    lineForm = line.split(split_symble)
    featureNumbers = len(lineForm) # feature number of matrix
    lineNumbers = len(f.readlines()) + 1 # line number of matrix

    f = open(filename)
    index = 0
    matrix = zeros((lineNumbers, featureNumbers))
    for line in f.readlines():
        line = line.strip()
        lineForm = line.split(split_symble)
        matrix[index, :] = lineForm[:]
        index += 1
    return matrix
