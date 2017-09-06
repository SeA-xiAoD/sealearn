from sealearn.DecisionTree_ID3 import DecisionTree_ID3
from sealearn.txt2list import txt2list

tree = DecisionTree_ID3()
dataSet = txt2list('data/ID3/lenses.txt')
X = [example[:-1] for example in dataSet]
y = [example[-1] for example in dataSet]
tree.fit(X, y, ['age', 'prescript', 'astigmatic', 'tearRate'])
print(tree.predict(['pre','hyper','no','normal']))
print(tree.precision())
tree.drawDecisionTree()
print("This decision tree is based on ID3 algorithm and there is no",
"pruning among this algorithm, so it is proning to overfitting.",
"So the C45 and CART decision tree is more useful in almost application.")
