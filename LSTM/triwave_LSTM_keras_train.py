import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import os, sys
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
import numpy as np
import matplotlib.pylab as plt

def _load_data(data, n_prev = 100):
    """
    data should be pd.DataFrame()
    """

    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data[i:i+n_prev])
        docY.append(data[i+n_prev])
        #docX.append(data.iloc[i:i+n_prev].as_matrix())
        #docY.append(data.iloc[i+n_prev].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY

def train_test_split(df, test_size=0.1):
    """
    This just splits data to training and testing parts
    """
    ntrn = round(len(df) * (1 - test_size))

    X_train, y_train = _load_data(df[0:ntrn,:])
    X_test, y_test = _load_data(df[ntrn:,:])

    return (X_train, y_train), (X_test, y_test)

print("python:{}, keras:{}, tensorflow: {}".format(sys.version, keras.__version__, tf.__version__))

# Define model
in_out_neurons = 2  
hidden_neurons = 300

model = Sequential()
model.add(LSTM(hidden_neurons, return_sequences=False,
               input_shape=(None, in_out_neurons)))
model.add(Dense(in_out_neurons, input_dim=hidden_neurons))  
model.add(Activation("linear"))  
model.compile(loss="mean_squared_error", optimizer="rmsprop")  


# Summary
model.summary()


# Create data
import pandas as pd
from random import random
flow = (list(range(1,10,1)) + list(range(10,1,-1)))*1000  
pdata = pd.DataFrame({"a":flow, "b":flow})  
pdata.b = pdata.b.shift(9)
data=pdata.iloc[10:]
#data = pdata.iloc[10:] * random()  # some noise


# Convert to numpy array
data=data.values
# Extract data from data frame
(X_train, y_train), (X_test, y_test) = train_test_split(data)  # retrieve data



# plt.figure(1)
# plt.clf()
# plt.plot(np.squeeze(X_train[0,:,:]))
# plt.plot(np.squeeze(X_train[1,:,:]),'--')
# plt.plot(np.squeeze(X_train[2,:,:]),'-o')


# and now train the model
# batch_size should be appropriate to your memory size
# number of epochs should be higher for real world problems
model.fit(X_train, y_train, batch_size=450, epochs=10, validation_split=0.05)  


# Make predictions
predicted = model.predict(X_test)  
rmse = np.sqrt(((predicted - y_test) ** 2).mean(axis=0))
print('RMSE on test data: {}'.format(rmse))

plt.rcParams["figure.figsize"] = (17, 9)
plt.plot(predicted[:100][:,0],"--")
plt.plot(predicted[:100][:,1],"--")
plt.plot(y_test[:100][:,0],":")
plt.plot(y_test[:100][:,1],":")
plt.legend(["Prediction 0", "Prediction 1", "Test 0", "Test 1"])
plt.show()

# Save Model
model_fname='tri_wave_keras.h5'
print('Saving file as %s'  % model_fname)
model.save(model_fname)
