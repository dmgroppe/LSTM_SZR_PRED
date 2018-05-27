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
import scipy.stats as stats

# np.random.seed(0)
# print('Seeding random seed generator!!!')

def norm_eeg(eeg):
    md=np.median(eeg)
    iqr=stats.iqr(eeg)
    print('Median={}, IQR={}'.format(md,iqr))
    shrinkage=3*iqr
    eeg=(eeg-md)/(shrinkage)
    return eeg, (md, shrinkage)


def gen_sim_art_eeg_data(art_type):
    # Load data and split it into train, valid, & test
    in_fname='SAMPLE_DATA/clean_data_py_format.mat'
    mat=sio.loadmat(in_fname)
    eeg=mat['data'][4,:] # Channel Fz
    n_tpt=len(eeg)
    eeg=(eeg-np.mean(eeg))/(2*np.std(eeg))

    art = np.zeros(eeg.shape)
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
    raw = eeg + art
    # plt.figure(1)
    # plt.plot(art)
    # plt.show()
    #
    # plt.figure(2)
    # plt.plot(art,label='art')
    # plt.plot(eeg,label='eeg')
    # plt.plot(raw,label='raw')
    # plt.legend()
    # plt.show()
    return raw, eeg, art


def load_eeg_data():
    # Load and normalize data
    in_fname = 'SAMPLE_DATA/GRU_DATA/p3_epoched_blink_group.mat'
    mat = sio.loadmat(in_fname)

    # Get first chan and normalize
    raw = np.squeeze(mat['raw_eeg'][0, :, :])
    raw, raw_nrm_fact = norm_eeg(raw)
    clean = np.squeeze(mat['cleaned_eeg'][0, :, :])
    clean, clean_nrm_fact = norm_eeg(clean)
    art = np.squeeze(mat['blink_eeg'][0, :, :])
    art, art_nrm_fact = norm_eeg(art)

    return raw, clean, art, raw_nrm_fact, clean_nrm_fact, art_nrm_fact


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
    n_hidden=120
    model.add(GRU(n_hidden,
              input_shape=(n_wind, 1),
              batch_size=batch_size,
              stateful=stateful))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    rmsp = RMSprop(lr=0.000001)
    #rmsp = RMSprop(lr=0.0001)
    model.compile(loss='mse', optimizer=rmsp)
    model.summary()
    return model


def format_ep(raw_ep, clean_ep, art_ep, n_wind, n_tpt, mid_wind):
    # x_train = n_tpt-n_wind x n_wind x 1
    # y_train = n_tpt-n_wind x 2
    tsteps=n_tpt-n_wind
    x=np.zeros((tsteps,n_wind,1))
    y=np.zeros((tsteps,2))
    for a in range(tsteps):
        x[a,:,0]=raw_ep[a:a+n_wind]
        y[a, 0] = clean_ep[a+mid_wind]
        y[a, 1] = art_ep[a + mid_wind]
    return x, y

######## Start of Main Script ########

#raw, eeg, art=gen_sim_art_eeg_data('periodic')
#raw, eeg, art=gen_sim_art_eeg_data('rand')
raw, clean, art, raw_nrm_fact, clean_nrm_fact, art_nrm_fact=load_eeg_data()
n_tpt, n_epoch=raw.shape
print('Total # of obs=%d' % (n_tpt*n_epoch))

# Split Data into Train, Valid, & Test
n_train_ep=int(np.floor(n_epoch*.5))
n_valid_ep=int(np.floor(n_epoch*.25))
n_test_ep=n_epoch-n_valid_ep-n_train_ep
print('%d training epochs, %f of data' % (n_train_ep, n_train_ep/n_epoch))
print('%d validation epochs, %f of data' % (n_valid_ep, n_valid_ep/n_epoch))
print('%d test epochs, %f of data' % (n_test_ep, n_test_ep/n_epoch))

train_ids=np.arange(0,n_train_ep)
valid_ids=np.arange(n_train_ep,n_train_ep+n_valid_ep)
test_ids=np.arange(n_train_ep+n_valid_ep,n_train_ep+n_valid_ep+n_test_ep)

print('Creating Stateful Model...')
batch_size = 1
n_wind=31
mid_wind=int(np.ceil(n_wind/2))
n_train_iter = 25 # max # of training iterations
n_train_batch=100 # # of epochs to train in each iteration on before computing validation error
n_valid_batch=500 # # of epochs to use for estimating validation error
n_test_batch=100 # # of test epochs to plot and estimate testing error on
stateful=True
model_stateful = create_model(stateful,n_wind,batch_size)
model_fname='temp_epoched.h5'

val_loss=list()
val_art_loss=list()
val_eeg_loss=list()
train_loss=list()
patience=3 # TODO make 3

# val_loss=np.zeros(epochs) # TODO
# train_loss=np.zeros(epochs)
print('Training')
n_last_improvement=0
best_val_loss=np.nan
for i in range(n_train_iter):
    print('Train iteration', i + 1, '/', n_train_iter)

    # Train on a few random epochs
    temp_loss=0
    for j in range(n_train_batch):
        # grab a random training epoch
        epoch_id=np.random.randint(0,n_train_ep)
        x_train, y_train=format_ep(raw[:,epoch_id], clean[:,epoch_id], art[:,epoch_id], n_wind, n_tpt, mid_wind)
        train_hist=model_stateful.fit(x_train,
                           y_train,
                           batch_size=batch_size,
                           epochs=1,
                           verbose=1,
                           shuffle=False)
        temp_loss+=train_hist.history['loss'][0]
        model_stateful.reset_states()
    train_loss.append(temp_loss/n_train_batch)

    #Estimate validation error from a few random epochs
    temp_art_loss = 0
    temp_eeg_loss = 0
    print('Estimating validation error from %d epochs' % n_valid_batch)
    for j in range(n_valid_batch):
        if (j%10)==0:
            print('Validation epoch %d/%d' % (j,n_valid_batch))
        # grab a random validation epoch
        epoch_id = np.random.randint(n_train_ep, n_train_ep+n_valid_ep)
        x, y = format_ep(raw[:, epoch_id], clean[:, epoch_id], art[:, epoch_id], n_wind, n_tpt, mid_wind)
        y_hat = model_stateful.predict(x,batch_size=batch_size)
        temp_eeg_loss += np.sqrt(np.mean(np.square(y[:,0]-y_hat[:,0])))
        temp_art_loss += np.sqrt(np.mean(np.square(y[:,1]-y_hat[:,1])))
        model_stateful.reset_states()

    val_eeg_loss.append(temp_eeg_loss / n_valid_batch)
    val_art_loss.append(temp_art_loss / n_valid_batch)
    val_loss.append((val_eeg_loss[-1]+val_art_loss[-1])/2)
    print('Valid error EEG=%f, Art=%f Both=%f' % (val_eeg_loss[-1],val_art_loss[-1],val_loss[-1]))
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



# Plot Training & Validation Error
plt.figure(1)
plt.clf()
plt.plot(np.asarray(val_loss),'-o',label='val')
plt.plot(np.asarray(val_art_loss),'-o',label='val_art')
plt.plot(np.asarray(val_eeg_loss),'-o',label='val_eeg')
plt.plot(np.asarray(train_loss),'-o',label='train')
plt.legend()
#plt.show()
plt.savefig('PICS/epoched_eeg_loss.pdf')
print('Best valid loss {}'.format(np.min(val_loss)))


print('Loading best model: %s' % model_fname)
model_stateful=load_model(model_fname)

# Estimate test set error from a few random epochs
temp_art_loss = 0
temp_eeg_loss = 0
print('Estimating test error from %d epochs' % n_test_batch)
for j in range(n_test_batch):
    if (j % 10) == 0:
        print('Testing epoch %d/%d' % (j, n_valid_batch))
    # grab a random validation epoch
    epoch_id = np.random.randint(n_train_ep+n_valid_ep, n_train_ep + n_valid_ep+n_test_ep)
    x, y = format_ep(raw[:, epoch_id], clean[:, epoch_id], art[:, epoch_id], n_wind, n_tpt, mid_wind)
    y_hat = model_stateful.predict(x, batch_size=batch_size)
    temp_eeg_loss += np.sqrt(np.mean(np.square(y[:, 0] - y_hat[:, 0])))
    temp_art_loss += np.sqrt(np.mean(np.square(y[:, 1] - y_hat[:, 1])))
    model_stateful.reset_states()
    plt.figure(2)
    plt.clf()
    for a in range(2):
        plt.subplot(1,2,a+1)
        plt.plot(y[:,a],'-b',label='y')
        plt.plot(y_hat[:, a],'-r',label='yhat')
        if a==0:
            plt.title('EEG, Epoch %d' % epoch_id)
        else:
            plt.title('Artifact, Epoch %d' % epoch_id)
        plt.legend()
    #plt.show()
    plt.savefig('PICS/epoched_eeg_yhat'+str(j)+'.pdf')
test_eeg_loss=temp_eeg_loss/n_test_batch
test_art_loss=temp_art_loss/n_test_batch
print('** Test Data Loss **')
print('EEG %f' % test_eeg_loss)
print('Art %f' % test_art_loss)
test_loss=(test_eeg_loss+test_art_loss)/2
print('Both %f' % test_loss)


