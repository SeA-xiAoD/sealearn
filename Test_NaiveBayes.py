from sealearn.NaiveBayes import NaiveBayes
import re
import random

def textParse(text):
    listOfToken = re.split(r'\W*', text)
    return [tok.lower() for tok in listOfToken if len(tok) > 2]

model = NaiveBayes()
# input email
email_list = []
label_list = []
for i in range(1, 26):
    email_list.append(textParse(open('data/NaiveBayes/ham/%d.txt' % i).read()))
    label_list.append(0)
    email_list.append(textParse(open('data/NaiveBayes/spam/%d.txt' % i).read()))
    label_list.append(1)

model.fit(email_list, label_list)
print(model.precision())
