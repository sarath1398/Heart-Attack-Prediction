import numpy as np 
import pandas as pd 
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import BaseDiscreteNB
from sklearn.naive_bayes import BaseNB
from sklearn.naive_bayes import _BaseDiscreteNB
from sklearn.naive_bayes import _BaseNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

f = pd.read_csv(r"C:\Users\Sarath\Desktop\DS Hackathon\Files\heart.csv")
f = np.asarray(f)
#print(f.shape)
x = f[:, 0:13]
#print(x)
y = f[:, 13]
#print(y)

#BernoulliNB
r = BernoulliNB()
r.fit(x, y)
y1 = r.predict(x)
print("BernoulliNB: ",accuracy_score(y, y1)) # 83.16% Accuracy 

#ComplementNB
r = ComplementNB()
r.fit(x, y)
y1 = r.predict(x)
print("ComplementNB: ", accuracy_score(y, y1)) # 75.24% Accuracy

#CategoricalNB
r = CategoricalNB()
r.fit(x, y)
y1 = r.predict(x)
print("CategoricalNB: ", accuracy_score(y, y1)) # 91.08% Accuracy
'''
#BaseDiscreteNB
r = BaseDiscreteNB()
r.fit(x, y)
y1 = r.predict(x)
print("Base Discrete NB: ", accuracy_score(y, y1))  

#BaseNB
r = BaseNB()
r.fit(x, y)
y1 = r.predict(x)
print("Base NB: ", accuracy_score(y, y1))

#_BaseDiscreteNB
r =_BaseDiscreteNB()
r.fit(x, y)
y1 = r.predict(x)
print("_Base Discrete NB: ", accuracy_score(y, y1))

#_BaseNB
r = _BaseNB()
r.fit(x, y)
y1 = r.predict(x)
print("_Base NB: ", accuracy_score(y, y1))
'''
#MultinomialNB
r = MultinomialNB()
r.fit(x, y)
y1 = r.predict(x)
print("MultinomialNB: ",accuracy_score(y, y1))  # 75.24% Accuracy

r = GaussianNB()
r.fit(x, y)
y1 = r.predict(x)
print("GaussianNB: ",accuracy_score(y, y1))  # 84.15% accuracy
