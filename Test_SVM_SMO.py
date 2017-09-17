from sealearn.SVM_SMO import SVM_SMO
from sealearn.txt2matrix import txt2matrix

model = SVM_SMO()

data = txt2matrix('data/SVM/testSet.txt')
X = data[:, :-1]
y = data[:, -1]

model.fit(X, y)
print(model.predict([3.018896,2.556416]))
print(model.precision())
