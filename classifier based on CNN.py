import tflearn
from tflearn.data_utils import shuffle,to_categorical
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization
import scipy.io as sio
import numpy as np
import pandas as pd
import tensorboard

resource='Enter your data mat format file path'
data_struct=sio.loadmat(resource)
X_train=data_struct['train_data']
X_train=X_train.reshape([515,3,30,7])
Y_train=data_struct['train_labels']
X_validation=data_struct['validation_data']
X_validation=X_validation.reshape([103,3,30,7])
Y_validation=data_struct['validation_labels']
X_test=data_struct['test_data']
Y_test=data_struct['validation_data']
X_test=X_test.reshape([103,3,30,7])

network=input_data(shape=[None,3,30,7])
network=conv_2d(network,60,3,padding='same',activation='relu')
network=conv_2d(network,132,3,padding='same',activation='relu')
network=local_response_normalization(network)
metwork=max_pool_2d(network,3)
network=fully_connected(network,120,activation='tanh',regularizer='L2', weight_decay=0.0012)
network=dropout(network,0.5)
network=fully_connected(network,48,activation='tanh',regularizer='L2', weight_decay=0.0012)
network=dropout(network,0.5)
network=fully_connected(network,4,activation='softmax')
network=regression(network,optimizer='adam',loss='categorical_crossentropy',learning_rate=0.0005)
model=tflearn.DNN(network,tensorboard_verbose=3,tensorboard_dir='')
model.fit(X_train,Y_train,n_epoch=80,shuffle=True,validation_set=(X_validation,Y_validation),show_metric=True,batch_size=18,run_id='RHEP_classifier')
model.save('my_model')
model.load('my_model')

result_train=model.predict(X_train)
print(result_train,type(result_train))
dataFrame=pd.DataFrame(result_train)
with pd.ExcelWriter('result_train.xlsx') as writer:
    dataFrame.to_excel(writer, sheet_name='page1', float_format='%.8f')

result_validation=model.predict(X_validation)
print(result_validation,type(result_validation))
dataFrame=pd.DataFrame(result_validation)
with pd.ExcelWriter('result_validation.xlsx') as writer:
    dataFrame.to_excel(writer, sheet_name='page1', float_format='%.8f')

result_test=model.predict(X_test)
print(result_test,type(result_test))
dataFrame=pd.DataFrame(result_test)
with pd.ExcelWriter('result_test.xlsx') as writer:
    dataFrame.to_excel(writer, sheet_name='page1', float_format='%.8f')



