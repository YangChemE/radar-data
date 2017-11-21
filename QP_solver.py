import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cvxopt
import cvxopt.solvers
from cvxopt import solvers

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data", header = None)
labels = df[df.columns[34]]
datas = df[df.columns[0:34]]


#encoding labels into numerical values
enc = LabelEncoder()
enc.fit(labels)
labels = enc.transform(labels)
        
datas = np.mat(datas)
labels = (np.mat(labels)).T

for i in range (len(df)):
    if (np.asscalar(labels[i,:]) == 0):
        labels[i,:] = -1

df = pd.DataFrame(np.column_stack((datas,labels)))

train, test = train_test_split(df,train_size = 0.5)



X_train = np.mat(train[train.columns[0:34]])
y_train = np.mat(train[train.columns[34]]).T

X_test = np.mat(test[test.columns[0:34]])
y_test = np.mat(test[test.columns[34]]).T

size_train, dim_train = train.shape


#formulating the inputs for QP solver
h = -(np.mat(np.ones((size_train,1))))
q = np.mat(np.zeros((35,1)))
p = np.eye(35)
p[34,34] = 0

G = y_train
row = X_train[0,:]*np.asscalar(y_train[0,:])
for i in range (1,size_train):
    y = np.asscalar(y_train[i,:])
    temp = (X_train[i,:] * y)
    row = np.mat(np.vstack((row,temp)))
G = -(np.mat(np.column_stack((row,G))))

#transfer the inputs to the required format
P = cvxopt.matrix(p,tc='d')
q = cvxopt.matrix(q,tc='d')
G = cvxopt.matrix(G,tc='d')
h = cvxopt.matrix(h,tc='d')

#solve the QP
sol = solvers.qp(P,q,G,h)

#get the weight vector and intercept from the QP result
w = np.mat(sol['x'])
b = np.asscalar(w[34,:])
w = w [0:34,:]

#Classifaction of the train set
X_train = np.mat(X_train)
y_predict_train = (X_train*w)+b
for i in range (len(y_predict_train)):
    yi = np.asscalar(y_predict_train[i,:])
    if (yi >= 0):
        y_predict_train[i,:] = 1
    elif (yi < 0):
        y_predict_train[i,:] = -1

#Prediction of the test set       
y_predict_test = (X_test*w)+b
for i in range (len(y_predict_test)):
    yi = np.asscalar(y_predict_test[i,:])
    if (yi >= 0):
        y_predict_test[i,:] = 1
    elif (yi < 0):
        y_predict_test[i,:] = -1

accuracy_train = ((len(train))-np.count_nonzero(y_train - y_predict_train))/(len(train))
accuracy_test = ((len(test))-np.count_nonzero(y_test - y_predict_test))/(len(test))

correct_train = (len(train))-np.count_nonzero(y_train - y_predict_train)
correct_test = (len(test))-np.count_nonzero(y_test - y_predict_test)

print(correct_train, " out of ", len(train), " were classified correctly in the train set")
print(correct_test, " out of ", len(test), " were classified correctly in the test set")
print("The model can classify the data withe accuracy of ", accuracy_train, " for the training set, and ", accuracy_test, " for the test set.")
