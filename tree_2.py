import scipy.io
from sklearn import tree
import cPickle
import csv


Data=[]
A=[]
for i in range(1,150):
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

#cPickle.dump(Data, open('D:\\Epilepsy\\save.p', 'wb')) 

res = tree.DecisionTreeClassifier()
res= res.fit(Data, A)

Test=[]
for i in range(1,503):
    addr='D:\\Epilepsy\\Dog_1\\Dog_1_test_segment_0'+str(str(i).zfill(3))+'.mat'
    eegDict= scipy.io.loadmat(addr)
    seg='test_segment_'+str(i)
    Test.append(eegDict[seg][0][0][0].ravel())
    A.append(1)
    print(addr)
#cPickle.dump(res, open('D:\\Epilepsy\\model.p', 'wb')) 

final_result=res.predict(Test)

with open('D:\\Epilepsy\\result.csv', 'w') as csvfile:
    fieldnames = ['clip', 'preictal']
    writer = csv.DictWriter(csvfile, lineterminator='\n', fieldnames=fieldnames)
    
    writer.writeheader()
    for i in range(503):
        addr='Dog_1_test_segment_0'+str(str(i).zfill(3))+'.mat'
        writer.writerow({'clip': addr, 'preictal': final_result[i]})
#with open('D:\\Epilepsy\\Data.txt', 'w') as f:
    #for s in Data:
        #f.write((str(s) + u'\n').encode('unicode-escape'))