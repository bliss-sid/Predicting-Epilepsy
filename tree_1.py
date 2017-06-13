import scipy.io
from sklearn import tree
import cPickle
import numpy as np

#eegDict=[]
Data=[]
A=[]
for i in range(1,400):
    addr='D:\\Epilepsy\\Dog_1\\Dog_1_interictal_segment_0'+str(str(i).zfill(3))+'.mat'
    eegDict= scipy.io.loadmat(addr)
    seg='interictal_segment_'+str(i)
    Data.append(eegDict[seg][0][0][0].ravel())
    A.append(0)
    print(addr)
for i in range(1,25):
    addr='D:\\Epilepsy\\Dog_1\\Dog_1_preictal_segment_0'+str(str(i).zfill(3))+'.mat'
    eegDict= scipy.io.loadmat(addr)
    seg='preictal_segment_'+str(i)
    Data.append(eegDict[seg][0][0][0].ravel())
    A.append(1)
    print(addr)

cPickle.dump(Data, open('D:\\Epilepsy\\save.p', 'wb')) 


res = tree.DecisionTreeClassifier()
res= res.fit(Data, A) #Training the model


print(res.predict(S))

print("Yes")