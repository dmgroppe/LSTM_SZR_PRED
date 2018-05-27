""" Word up. """

from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, GRU, Dropout, Activation
#from keras.layers import Dense, LSTM, GRU
from keras.models import load_model
from keras.optimizers import RMSprop
import scipy.io as sio


np.random.seed(0)

def gen_sim_art_eeg_data(art_type):
    # Load data and split it into train, valid, & test
    in_fname='SAMPLE_DATA/clean_data_py_format.mat'
    mat=sio.loadmat(in_fname)
    eeg=mat['data'][4,:] # Channel Fz
    n_tpt=len(eeg)
    eeg=(eeg-np.mean(eeg))/(2*np.std(eeg))

    art = np.zeros(eeg.shape)
    if art_type=='periodic':
        # Create simulated triangle wave artifact
        delt=.1
        for t in range(1,n_tpt):
            if (art[t-1]>=1) or (art[t-1]<=-1):
                delt=-delt
            art[t]=art[t-1]+delt
        art=art+1
        # scale some more TODO edit later
        eeg = eeg * .3
        art = art * .7
        dirty_eeg = eeg + art
    else:
        # Create triangle waves at random intervals
        # blinks last about 60 samples (~0.5 seconds)
        in_art=False
        delt=.05
        for t in range(1,n_tpt):
            if in_art==False:
                if np.random.rand(1)[0]>=0.999:
                    in_art=True
                    art_val = 0
                    ct = 0
            elif ct<=60:
                art_val=art_val+delt
                art[t]=np.sin(art_val)
                ct=ct+1
            else:
                in_art=False
        # scale some more TODO edit later
        eeg = eeg * .1
        art = art * .7
        dirty_eeg = eeg + art
        # plt.figure(1)
        # plt.plot(art)
        # plt.show()
        #
        # plt.figure(2)
        # plt.plot(art,label='art')
        # plt.plot(eeg,label='eeg')
        # plt.plot(dirty_eeg,label='dirty_eeg')
        # plt.legend()
        # plt.show()

    return dirty_eeg, eeg, art


def format_eeg_data():
    # Load data and split it into train, valid, & test
    in_fname='SAMPLE_DATA/clean_data_py_format.mat'
    mat=sio.loadmat(in_fname)
    eeg=mat['data'][4,:] # Channel Fz
    eeg=(eeg-np.mean(eeg))/(2*np.std(eeg))

    in_fname='SAMPLE_DATA/raw_data_py_format.mat'
    mat=sio.loadmat(in_fname)
    dirty_eeg=mat['data'][4,:] # Channel Fz
    dirty_eeg=(dirty_eeg-np.mean(dirty_eeg))/(2*np.std(dirty_eeg))

    in_fname='SAMPLE_DATA/raw_data_py_format.mat'
    mat=sio.loadmat(in_fname)
    dirty_eeg=mat['data'][4,:] # Channel Fz
    dirty_eeg=(dirty_eeg-np.mean(dirty_eeg))/(2*np.std(dirty_eeg))

    art=dirty_eeg-eeg

    return dirty_eeg, eeg, art


def get_data_clip(x,y_eeg,y_art,id_range,n_clip_tpt,n_wind):
    temp_n_tpt=len(x)
    x_clip=np.zeros((n_clip_tpt,n_wind,1))
    y_clip=np.zeros((n_clip_tpt,2))
    start_id=np.random.randint(id_range[0],id_range[1]-n_clip_tpt-n_wind)
    #start_id=np.random.randint(0,temp_n_tpt-n_clip_tpt)
    #start_id=0
    print('start_id: {}'.format(start_id))
    for ct, cursor in enumerate(range(start_id,start_id+n_clip_tpt)):
        x_clip[ct,:,0]=x[cursor:cursor+n_wind]
        y_clip[ct,0]=y_eeg[cursor+int(np.floor(n_wind/2))] # Target output is the value of the clean and artifact data at
        # the center time point
        y_clip[ct,1]=y_art[cursor+int(np.floor(n_wind/2))]
        #x_clip[ct,:,0]=x[0,cursor:cursor+n_wind]
        #y_clip[ct,0]=y[0,cursor+int(np.floor(n_wind/2))]
    x_clip=np.flip(x_clip,1) #TODO do this once to the data to speed up clip generation
    return x_clip, y_clip


def get_all_data(x,y_eeg,y_art,id_range,n_wind):
    temp_n_tpt=id_range[1]-id_range[0]
    n_clip_tpt=temp_n_tpt-n_wind-1
    x_clip=np.zeros((n_clip_tpt,n_wind,1))
    y_clip=np.zeros((n_clip_tpt,2))
    for ct in range(0,n_clip_tpt):
        x_clip[ct,:,0]=x[id_range[0]+ct:id_range[0]+ct+n_wind]
        y_clip[ct,0]=y_eeg[id_range[0]+ct+int(np.floor(n_wind/2))] # Target output is the value of the clean and artifact data at
        # the center time point
        y_clip[ct,1]=y_art[id_range[0]+ct+int(np.floor(n_wind/2))]
        #x_clip[ct,:,0]=x[0,cursor:cursor+n_wind]
        #y_clip[ct,0]=y[0,cursor+int(np.floor(n_wind/2))]
    x_clip=np.flip(x_clip,1) #TODO do this once to the data to speed up clip generation
    return x_clip, y_clip


def normalize(x):
    sd=np.std(x)*2
    mn=np.mean(x)
    x=(x-mn)/sd
    return x, mn, sd


def inv_normalize(x,mn,sd):
    return x*sd+mn


# Function for creating model
def create_model(stateful,n_wind,batch_size):
    model = Sequential()
    n_hidden=240
    model.add(GRU(n_hidden,
              input_shape=(n_wind, 1),
              batch_size=batch_size,
              stateful=stateful))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    # rmsp = RMSprop(lr=0.000001)
    rmsp = RMSprop(lr=0.0001)
    model.compile(loss='mse', optimizer=rmsp)
    model.summary()
    return model

#dirty_eeg, eeg, art=gen_sim_art_eeg_data('periodic')
#dirty_eeg, eeg, art=gen_sim_art_eeg_data('rand')
dirty_eeg, eeg, art=format_eeg_data()

n_tpt=len(eeg)
dirty_eeg, dirty_eeg_mn, dirty_eeg_sd=normalize(dirty_eeg)
eeg, eeg_mn, eeg_sd=normalize(eeg)
art, art_mn, art_sd=normalize(art)


# Split Data into Train, Valid, & Test
n_train_tpt=int(np.floor(n_tpt*.5))
n_valid_tpt=int(np.floor(n_tpt*.25))
n_test_tpt=n_tpt-n_valid_tpt-n_train_tpt
print('%d training tpts, %f of data' % (n_train_tpt, n_train_tpt/n_tpt))
print('%d validation tpts, %f of data' % (n_valid_tpt, n_valid_tpt/n_tpt))
print('%d test tpts, %f of data' % (n_test_tpt, n_test_tpt/n_tpt))

train_ids=(0,n_train_tpt)
valid_ids=(n_train_tpt,n_train_tpt+n_valid_tpt)
test_ids=(n_train_tpt+n_valid_tpt,n_train_tpt+n_valid_tpt+n_test_tpt)


print('Creating Stateful Model...')
batch_size = 1
n_wind=61
epochs = 25
stateful=True
model_stateful = create_model(stateful,n_wind,batch_size)
model_fname='temp_stateful.h5'

val_loss=list()
train_loss=list()
patience=3


# val_loss=np.zeros(epochs)
# train_loss=np.zeros(epochs)
print('Training')
n_last_improvement=0
best_val_loss=np.nan
x_valid_clip, y_valid_clip = get_all_data(dirty_eeg, art, eeg, valid_ids, n_wind)
for i in range(epochs):
    print('Epoch', i + 1, '/', epochs)
    # Note that the last state for sample i in a batch will
    # be used as initial state for sample i in the next batch.
    # Thus we are simultaneously training on batch_size series with
    # lower resolution than the original series contained in data_input.
    # Each of these series are offset by one step and can be
    # extracted with data_input[i::batch_size].
    n_clip_tpt=120*10
    #x_train_clip, y_train_clip=get_data_clip(x_train,y_train,n_clip_tpt,n_wind)
    x_train_clip, y_train_clip=get_data_clip(dirty_eeg,art,eeg,train_ids,n_clip_tpt,n_wind)
    # print(x_train_clip.shape)
    # print(y_train_clip.shape)
    #exit()
    #n_clip_tpt=120*100
    # x_valid_clip, y_valid_clip=get_data_clip(dirty_eeg,art,eeg,valid_ids,n_clip_tpt,n_wind)
    train_hist_stateful=model_stateful.fit(x_train_clip,
                       y_train_clip,
                       batch_size=batch_size,
                       epochs=1,
                       verbose=1,
                       validation_data=(x_valid_clip, y_valid_clip),
                       shuffle=False)
    val_loss.append(train_hist_stateful.history['val_loss'][0])
    train_loss.append(train_hist_stateful.history['loss'][0])
    if np.isnan(best_val_loss) or best_val_loss>val_loss[-1]:
        # first epoch or improved performance
        best_val_loss=val_loss[-1]
        n_last_improvement=0
        print('Best validation loss so far. Saving model as %s' % model_fname)
        model_stateful.save(model_fname)
    else:
        # No improvment
        n_last_improvement+=1
    if n_last_improvement>=patience:
        print('Validation loss not improving. EXITING!')
        break
    model_stateful.reset_states()



plt.figure(1)
plt.clf()
plt.plot(np.asarray(val_loss),'-o',label='val')
plt.plot(np.asarray(train_loss),'-o',label='train')
plt.legend()
plt.show()
print('Best valid loss {}'.format(np.min(val_loss)))


print('Loading best model: %s' % model_fname)
model_stateful=load_model(model_fname)

#n_clip_tpt=1000
#x_test_clip, y_test_clip=get_data_clip(dirty_eeg,art,eeg,test_ids,n_clip_tpt,n_wind)
x_test_clip, y_test_clip=get_all_data(dirty_eeg,art,eeg,test_ids,n_wind)

print('Predicting')
predicted_stateful = model_stateful.predict(x_test_clip, batch_size=batch_size)
print('Done')



print('Plotting Results')
plt.figure(1)
plt.clf()
for a in range(2):
    plt.subplot(2,1,a+1)
    plt.plot(y_test_clip[:,a],label='y')
    plt.plot(predicted_stateful[:,a],label='yhat')
    plt.legend()
plt.show()

temp_dirty_eeg=np.zeros(x_test_clip.shape[0])
temp_clean_eeg=np.zeros(x_test_clip.shape[0])
temp_art=np.zeros(x_test_clip.shape[0])
for a in range(len(temp_dirty_eeg)):
    temp_dirty_eeg[a]=x_test_clip[a,3,0]
    temp_clean_eeg[a] = y_test_clip[a,1]
    temp_art[a] = y_test_clip[a, 0]
temp_dirty_eeg=inv_normalize(temp_dirty_eeg,dirty_eeg_mn,dirty_eeg_sd)
temp_clean_eeg=inv_normalize(temp_clean_eeg,eeg_mn,eeg_sd)
temp_art=inv_normalize(temp_art,art_mn,art_sd)
temp_art_hat=inv_normalize(predicted_stateful[:,0],art_mn,art_sd)

n_show=200
plt.figure(2)
plt.clf()
for a in range(2):
    plt.subplot(3,1,a+1)
    plt.plot(y_test_clip[:n_show,a],label='y')
    plt.plot(predicted_stateful[:n_show,a],label='yhat')
    plt.legend()
plt.subplot(3,1,3)
plt.plot(temp_clean_eeg[:n_show],label='clean')
plt.plot(temp_dirty_eeg[:n_show],label='dirty')
plt.plot(temp_dirty_eeg[:n_show]-temp_art_hat[:n_show],label='recon')
# plt.plot(temp_dirty_eeg[:n_show]-temp_art[:n_show],'o',label='recon eeg')
#plt.plot(temp_dirty_eeg[:n_show]-predicted_stateful[:n_show,0],label='recon eeg')
plt.legend()
plt.show()
