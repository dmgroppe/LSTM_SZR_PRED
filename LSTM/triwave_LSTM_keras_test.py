import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import os, sys
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.models import load_model
import numpy as np
import matplotlib.pylab as plt
#import keras.backend as K

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

# Create data
import pandas as pd
from random import random
flow = (list(range(1,10,1)) + list(range(10,1,-1)))*1000
pdata = pd.DataFrame({"a":flow, "b":flow})
pdata.b = pdata.b.shift(9)
data = pdata.iloc[10:]
#data = pdata.iloc[10:] * random()  # some noise


# Convert to numpy array
data=data.values
# Extract data from data frame
(X_train, y_train), (X_test, y_test) = train_test_split(data)  # retrieve data


# Load Model and Generate Predictions
model2=load_model('tri_wave_keras.h5')
predicted2 = model2.predict(X_test)

# Summary
model2.summary()

# Plot actual and predicted data
plt.figure(1)
plt.rcParams["figure.figsize"] = (17, 9)
plt.plot(predicted2[:100][:,0],"--")
plt.plot(predicted2[:100][:,1],"--")
plt.plot(y_test[:100][:,0],":")
plt.plot(y_test[:100][:,1],":")
plt.legend(["Prediction 0", "Prediction 1", "Test 0", "Test 1"])
plt.show()

# Plot actual and predicted data
plt.figure(2)
plt.rcParams["figure.figsize"] = (17, 9)
plt.plot(predicted2[-100:][:,0],"--")
plt.plot(predicted2[-100:][:,1],"--")
plt.plot(y_test[-100:][:,0],":")
plt.plot(y_test[-100:][:,1],":")
plt.legend(["Prediction 0", "Prediction 1", "Test 0", "Test 1"])
plt.show()