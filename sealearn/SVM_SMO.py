from numpy import *

class optStruct:
    '''A structure using in computation.'''

    def __init__(self,dataMatIn, classLabels, C, toler):  # Initialize the structure with the parameters
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2))) #first column is valid flag

class SVM_SMO():
    '''A simple SVM using the SMO algorithm.'''

    def __init__(self):
        self.original_X = 0
        self.original_labels = 0
        self.b = 0
        self.alphas = 0
        self.ws = 0
        print('Initializ a SVM module using SMO algorithm.')

    def __calcEk(self, oS, k):
        fXk = float(multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T)) + oS.b
        return fXk - float(oS.labelMat[k])

    def __selectJrand(self, i,m):
        j=i #we want to select any J not equal to i
        while (j==i):
            j = int(random.uniform(0,m))
        return j

    def __clipAlpha(self, aj,H,L):
        if aj > H:
            aj = H
        if L > aj:
            aj = L
        return aj

    def __selectJ(self, i, oS, Ei):         #this is the second choice -heurstic, and calcs Ej
        maxK = -1; maxDeltaE = 0; Ej = 0
        oS.eCache[i] = [1,Ei]  #set valid #choose the alpha that gives the maximum delta E
        validEcacheList = nonzero(oS.eCache[:,0].A)[0]
        if (len(validEcacheList)) > 1:
            for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
                if k == i: continue #don't calc for i, waste of time
                Ek = self.__calcEk(oS, k)
                deltaE = abs(Ei - Ek)
                if (deltaE > maxDeltaE):
                    maxK = k; maxDeltaE = deltaE; Ej = Ek
            return maxK, Ej
        else:   #in this case (first time around) we don't have any valid eCache values
            j = self.__selectJrand(i, oS.m)
            Ej = self.__calcEk(oS, j)
        return j, Ej

    def __updateEk(self, oS, k):#after any alpha has changed update the new value in the cache
        Ek = self.__calcEk(oS, k)
        oS.eCache[k] = [1,Ek]

    def __innerL(self, i, oS):
        Ei = self.__calcEk(oS, i)
        if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
            j,Ej = self.__selectJ(i, oS, Ei) #this has been changed from selectJrand
            alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
            if (oS.labelMat[i] != oS.labelMat[j]):
                L = max(0, oS.alphas[j] - oS.alphas[i])
                H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
            else:
                L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
                H = min(oS.C, oS.alphas[j] + oS.alphas[i])
            if L==H:
                return 0
            eta = 2.0 * oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T
            if eta >= 0:
                return 0
            oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
            oS.alphas[j] = self.__clipAlpha(oS.alphas[j],H,L)
            self.__updateEk(oS, j) #added this for the Ecache
            if (abs(oS.alphas[j] - alphaJold) < 0.00001):
                 return 0
            oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
            self.__updateEk(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
            b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
            b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
            if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
            elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
            else: oS.b = (b1 + b2)/2.0
            return 1
        else: return 0

    def __smoPK(self, dataMatIn, classLabels, C, toler, maxIter):    #full Platt SMO
        oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler)
        iter = 0
        entireSet = True; alphaPairsChanged = 0
        while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
            alphaPairsChanged = 0
            if entireSet:   #go over all
                for i in range(oS.m):
                    alphaPairsChanged += self.__innerL(i,oS)
                iter += 1
            else:#go over non-bound (railed) alphas
                nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
                for i in nonBoundIs:
                    alphaPairsChanged += self.__innerL(i,oS)
                iter += 1
            if entireSet: entireSet = False #toggle entire set loop
            elif (alphaPairsChanged == 0): entireSet = True
        return oS.b,oS.alphas

    def __calcWs(self, alphas,dataArr,classLabels):
        X = mat(dataArr); labelMat = mat(classLabels).transpose()
        m,n = shape(X)
        w = zeros((n,1))
        for i in range(m):
            w += multiply(alphas[i]*labelMat[i],X[i,:].T)
        return w

    def fit(self, X, y):
        '''Input X and y.'''
        if len(X) != len(y):
            print('ERROR: The number of X and y is not match!')
            return
        self.original_X = X
        self.original_labels = y
        self.b, self.alphas = self.__smoPK(X, y, 0.6, 0.001, 40)
        self.ws = self.__calcWs(self.alphas, X, y)
        print(self.ws)
        print('The model fitting is finished!')

    def predict(self, new_X):
        '''Function using to predict the label of new input X.'''
        if self.ws.all() == 0:
            print('ERROR: The model is not fited!')
            return
        new_X = mat(new_X)
        if new_X * self.ws + self.b > 0:
            return 1
        else:
            return -1

    def correctRate(self):
        '''Function to predict labels of original input X, then using new labels
        to compare with original labels, and output correct rate of this model.'''
        if self.ws.all() == 0:
            print('ERROR: The model is not fited!')
            return
        correct_count = 0
        for i in range(0, len(self.original_X)):
            label = 1 if mat(self.original_X[i]) * self.ws + self.b > 0 else -1
            if label == self.original_labels[i]:
                correct_count += 1
        return correct_count / len(self.original_X) * 100
