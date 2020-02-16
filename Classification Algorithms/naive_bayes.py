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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

fitdata = pd.read_excel(r"C:\Users\Sarath\Desktop\DS Hackathon\Files\UCI Repository\fitdata.xlsx")
predictdata = pd.read_excel(r"C:\Users\Sarath\Desktop\DS Hackathon\Files\UCI Repository\predictiondata.xlsx")
fitdata = np.asarray(fitdata)
predictdata = np.asarray(predictdata)
X = fitdata[:, 0:13]
Y = fitdata[:, 13]
x = predictdata[:, 0:13]
y = predictdata[:, 13]
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)
x = scaler.fit_transform(x)
np.set_printoptions(precision=3)


#BernoulliNB
r = BernoulliNB()
r.fit(X, Y)
Y1 = r.predict(X)
Y2 = r.predict(x)
print(accuracy_score(Y, Y1)) # 83.8% Accuracy
print( accuracy_score(y, Y2))  # 83.8% Accuracy


#ComplementNB

r = ComplementNB()
r.fit(X, Y)
Y1 = r.predict(X)
Y2 = r.predict(x)
print(accuracy_score(Y, Y1))  
print(accuracy_score(y, Y2))  
'''
#CategoricalNB

r = CategoricalNB()
r.fit(X, Y)
Y1 = r.predict(X)
Y2 = r.predict(x)
print(accuracy_score(Y, Y1))  # 83.8% Accuracy
print(accuracy_score(y, Y2))  # 83.8% Accuracy
'''
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
r.fit(X, Y)
Y1 = r.predict(X)
Y2 = r.predict(x)
print(accuracy_score(Y, Y1))  
print(accuracy_score(y, Y2))  

#GaussianNB

r = GaussianNB()
r.fit(X, Y)
Y1 = r.predict(X)
Y2 = r.predict(x)
print(accuracy_score(Y, Y1))  # 83.8% Accuracy
print(accuracy_score(y, Y2))  # 83.8% Accuracy
