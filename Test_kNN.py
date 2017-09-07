from sealearn.kNN import kNN
from sealearn.txt2matrix import txt2matrix

s = kNN()

inputfile = 'data/kNN/datingTestSet.txt'
m = txt2matrix(inputfile)
X = m[:,:-1]
y = m[:,-1]

s.fit(X, y)
for i in range(1,20):
    print(s.precision(k = i))
print('We can see the precision is decreasing, because we use original to',
'predict has no sense due to the theory of kNN algorithm.',
'But it is a good way to test if our algorithm is running correctly.')
