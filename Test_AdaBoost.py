from sealearn.AdaBoost import AdaBoost
from sealearn.txt2matrix import txt2matrix

model = AdaBoost()
train_data = txt2matrix('data/AdaBoost/horseColicTraining2.txt')
train_X = train_data[:, :-1]
train_y = train_data[:, -1]

test_data = txt2matrix('data/AdaBoost/horseColicTest2.txt')
test_X = test_data[:, :-1]
test_y = test_data[:, -1]

model.fit(train_X, train_y)
print("Training data's correct rate:", model.correctRate())

correct_count = 0
for i in range(0, len(test_X)):
    label = model.predict(test_X[i])
    if label == test_y[i]:
        correct_count += 1
print("Test data's correct rate:", correct_count / len(test_X) * 100)
