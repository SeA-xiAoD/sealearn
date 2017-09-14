from sealearn.LogisticRegression import LogisticRegression
from sealearn.txt2matrix import txt2matrix

# test data set

model = LogisticRegression()

data = txt2matrix('data/LogisticRegression/testSet.txt')
X = data[:, :-1]
y = data[:, -1]
model.fit(X, y)
print('Test data set precision: ', model.precision())

# horse colic data set

horse_data_train = txt2matrix('data/LogisticRegression/horseColicTraining.txt')
horse_data_test = txt2matrix('data/LogisticRegression/horseColicTest.txt')

horse_data_train_X = horse_data_train[:, :-1]
horse_data_train_y = horse_data_train[:, -1]
horse_data_test_X = horse_data_test[:, :-1]
horse_data_test_y = horse_data_test[:, -1]

new_model = LogisticRegression()
new_model.fit(horse_data_train_X, horse_data_train_y)

correct_count = 0
for i in range(0, len(horse_data_test_X)):
    if new_model.predict(horse_data_test_X[i]) == horse_data_test_y[i]:
        correct_count += 1
print('Horse colic data set precision: ', correct_count / len(horse_data_test_X) * 100)
