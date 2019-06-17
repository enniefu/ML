from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
import numpy as np


digits = datasets.load_digits()
X=digits.data
y=digits.target

for aa in range(10):
    X_train, X_test, y_train_pre, y_test_pre = train_test_split(X, y, test_size=0.3)
    y_train=[]
    y_test=[]

    for i in y_train_pre:
        if i==aa:
            y_train.append(i)
        else:
            y_train.append(-1)

    for i in y_test_pre:
        if i==aa:
            y_test.append(i)
        else:
            y_test.append(-1)

    y_train=np.array(y_train)
    y_test=np.array(y_test)

    OCSVM=OneClassSVM(gamma=1,kernel='rbf')

    OCSVM.fit(X_train,y_train)

    ans=y_test-OCSVM.predict((X_test))

    i
    for a in ans:
        if a==0:
            i=i+1
    print(aa,":   ",i/len(ans))

