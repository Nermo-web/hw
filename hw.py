import numpy as np
import pandas as pd
import os

index_dict = {'running':1,'sitting':2,'standing':3,'walking':4} #the four classes's label
path = os.getcwd()
filenames = os.listdir(path)    #get the filenames in current directory

data_x = np.zeros((1,3))        #this process is to get the data
data_y = np.zeros((1))
for item in filenames:
    if '.csv' in item:
        tmp = pd.read_csv(item)
        tmp = tmp.drop(['id'],axis=1)
        tmp = np.array(tmp.values)
        data_x = np.concatenate([data_x,tmp],axis=0)
        look_y = item.split('_')[1]
        look_y = index_dict[look_y]
        tmp_y = np.array([look_y for _ in range(len(tmp))])
        data_y = np.concatenate([data_y,tmp_y])
        #data_y = np.concatenate([data_y,])
        # print(len(data_y),len(data_x))
        # print(look_y)
        # break

data_x = data_x[1:]
data_y = data_y[1:]
# print(len(data_x),len(data_y))
# print(data_x)


from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ratio = 0.99        #train and validation split ratio
split_t = int(len(data_x)*ratio)
data_x_train = data_x[:split_t]
data_x_val = data_x[split_t:]
data_y_train = data_y[:split_t]
data_y_val = data_y[split_t:]

model = MLPClassifier(activation='relu',max_iter=1000)
scaler = StandardScaler()       #preprocess the raw data to be zero mean and 1 variance
scaler.fit(data_x_train)
data_real_train = scaler.transform(data_x_train)
data_real_val = scaler.transform(data_x_val)

model.fit(data_real_train,data_y_train) #rain the data
print('the accuracy on the train set and validation set is: {} and {}'.format(model.score(data_real_train,data_y_train),\
                                                            model.score(data_real_val,data_y_val)))

