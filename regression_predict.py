#Importing Files 
from __future__ import print_function, division, absolute_import, unicode_literals
import csv
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model

csvf = open('C:\\Users\\siddharth\\Desktop\\GyanMatrix\\dataset.csv','rU') #Importing Dataset
rows = csv.reader(csvf)
game_data = [row for row in rows]
csvf.close()


#removing headers
game_data= game_data[1:]
game_data1=game_data
#Specifing clusters for Kmeans.Initially we use 2 clusters.
K=2
num_listings = len(game_data) 

#List of Game Names
game_names = []
for row in game_data:
    game_names.append(row[2])

gm_data=[]
colnum=[1,4,5,6,7,8,9,10]
encode_col=[1,4,6,7]
cl=0
ind=0
#print("Range of LabelEncoded Data according to columns\n")
#print("Format--> Col number   min element-max element   number of distinct elements\n") 
#Label Encoding to convert strings so that we can use data with fit function
for col in zip(*game_data): # Getting transpose to iterate through columns
    if(cl in colnum):
        if(cl in encode_col):
            lb = LabelEncoder()
            gm_data.append(lb.fit_transform(col))
        else:
            col1=[float(i) for i in col]
            gm_data.append(col1)
            
        #print(ind+1,"\t\t\t  ",min(gm_data[ind]),"-",max(gm_data[ind]),"\t\t  ",len(set(gm_data[ind])))
        ind+=1
    cl=cl+1

award=gm_data[4]
del gm_data[4]
#print(award)
game_data=zip(*gm_data)#Finally transposing to get our data


#We are using first 12000 listings for training the model and following for testing the model. 
train=game_data[:12000]
award_train=award[:12000]
test=game_data[12001:]
award_test=award[12001:]

res = linear_model.LinearRegression()

res = res.fit(train, award_train) #Training the model



total_counter=0
right_counter=0
result=res.predict(test)
#print(result)
#print(award_test)
for i in range(len(test)):
    total_counter+=1
    #print(test[i])
    #print(res.predict(test[i]))
    if(abs(result[i])>5):
        result[i]=1
    else:
        result[i]=0
    if(result[i]==award_test[i]):
        right_counter+=1
print("Accuracy-->",right_counter*100.0/total_counter)


#For predicting whether a game will win editor choice award or not-> Input its score_phrase,platform,score,genre,release_year,release_month,release_date

#For Example,lets create a general list containing these details about the game
A=(7, 33, 6.8, 76, 2016, 6.0, 28.0) #These value are encoded to fit in the model..Modelling of the entire data into this format is done above using LabelEncoder
#To predict call res.predict(A) after reshaping A to get rid of deprecation warning
A=np.array(A).reshape(1,-1)
if(abs(res.predict(A))>5):
    print("Y")
else:
    print("N")

