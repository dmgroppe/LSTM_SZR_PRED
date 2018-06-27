""" Word up. """
from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import sys
from keras.models import Sequential, load_model
from keras.layers import Dense, SimpleRNN, Activation, Dropout
if sys.platform=='darwin':
    from keras.layers import GRU
else:
    from keras.layers import CuDNNGRU, GRU
from keras.optimizers import RMSprop, SGD
from keras.constraints import max_norm
import scipy.io as sio
import scipy.stats as stats
import os

# import ieeg_funcs as ief
# import dgFuncs as dg
#import pickle
#from sklearn.externals import joblib
import json
from shutil import copyfile


# np.random.seed(0)
# print('Seeding random seed generator!!!')

def norm_eeg(eeg):
    md=np.median(eeg)
    iqr=stats.iqr(eeg)
    print('Median={}, IQR={}'.format(md,iqr))
    shrinkage=3*iqr
    eeg=(eeg-md)/(shrinkage)
    return eeg, (md, shrinkage)


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


# Function for creating model
def create_model(stateful,n_wind,batch_size,n_hidden,n_layers,lrate,opt):
    model = Sequential()
    # Add GRU Layer
    if n_layers==1:
        if sys.platform == "darwin":
            model.add(GRU(n_hidden,
                      input_shape=(n_wind, 1),
                      batch_size=batch_size,
                      return_sequences=False, # This needs to be false unless it is followed by another recurrent layer
                      stateful=stateful))
        else:
            model.add(CuDNNGRU(n_hidden,
                      input_shape=(n_wind, 1),
                      batch_size=batch_size,
                      return_sequences=False,
                      stateful=stateful))
    # Add RELU layer
    model.add(Dense(n_hidden))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # Linear Output Layer
    model.add(Dense(2))
    # Optimizer
    if opt=='sgd':
        print('Using SGD...')
        optim=SGD(lr=lrate, momentum=lrate/10, decay=0., nesterov=True)
    else:
        print('Using RMSprop...')
        optim = RMSprop(lr=lrate)
        #rmsp = RMSprop(lr=0.0001)
    model.compile(loss='mse', optimizer=optim)
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
if len(sys.argv)==1:
    print('Usage: train_eeg_gru_epoched.py train_params.json')
    exit()
if len(sys.argv)!=2:
    raise Exception('Error: train_eeg_gru_epoched.py requires 1 argument: train_params.json')

# Import Parameters from json file
param_fname=sys.argv[1]
print('Importing model parameters from %s' % param_fname)
with open(param_fname) as param_file:
    params=json.load(param_file)
model_name=params['model_name']
print('Model name is %s' % model_name)
n_layers=int(params['n_layers'])
print('n_layers is %s' % n_layers)
n_hidden=int(params['n_hidden'])
print('hidden units/layer is %d' % n_hidden)
patience=int(params['patience'])
print('patience is %d' % patience)
lr=float(params['learning_rate'])
print('learning rate is %f' % lr)
opt=params['optimizer']
print('learning rate is %s' % opt)

# Find if there are any existing models of this name
# If so, grab the number of model an increment that number by 1 to get new model name
if sys.platform=='darwin':
    model_root = '/Users/davidgroppe/PycharmProjects/LSTM_SZR_PRED/MODELS/'
else:
    model_root='/home/dgroppe/GIT/LSTM_SZR_PRED/MODELS/'
model_num=1
for f in os.listdir(model_root):
    if os.path.isdir(os.path.join(model_root,f)):
        spltf=f.split('_')
        if spltf[0]==model_name:
           temp_model_num=int(spltf[1])
           if temp_model_num>=model_num:
               model_num=temp_model_num+1
model_path=os.path.join(model_root,model_name+'_'+str(model_num))
print('Model will be stored to %s' % model_path)
if os.path.exists(model_path)==False:
    os.mkdir(model_path)
copyfile(param_fname,os.path.join(model_path,param_fname))
# Make folder for figures
fig_path = os.path.join(model_path, 'PICS')
os.mkdir(fig_path)

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
#n_hidden=np.zeros(2,dtype=int)
#n_hidden[0]=128
#n_hidden[1]=512
mid_wind=int(np.ceil(n_wind/2))
if True:
    n_train_iter = 200 # max # of training iterations
    n_train_batch=100 # # of epochs to train in each iteration on before computing validation error
    n_valid_batch=500 # # of epochs to use for estimating validation error
    n_test_batch=100 # # of test epochs to plot and estimate testing error on
else:
    # For debugging
    n_train_iter = 2  # max # of training iterations
    n_train_batch = 1  # # of epochs to train in each iteration on before computing validation error
    n_valid_batch = 1  # # of epochs to use for estimating validation error
    n_test_batch = 1  # # of test epochs to plot and estimate testing error on

stateful=True
model_stateful = create_model(stateful,n_wind,batch_size,n_hidden,n_layers,lr,opt)
model_fname='cuda_gru_epoched.h5'
out_metrics_fname = os.path.join(model_path, 'train_metrics.npz')

val_loss=list()
val_art_loss=list()
val_eeg_loss=list()
train_loss=list()
train_loss_se=list()
val_loss_se = list()
val_art_loss_se = list()
val_eeg_loss_se = list()

# val_loss=np.zeros(epochs) # TODO
# train_loss=np.zeros(epochs)
print('Training')
n_last_improvement=0
best_val_loss=np.nan
for i in range(n_train_iter):
    print('Train iteration', i + 1, '/', n_train_iter)

    # Train on a few random epochs
    temp_loss=np.zeros(n_train_batch)
    temp_loss_se = np.zeros(n_train_batch)
    for j in range(n_train_batch):
        if (j%10)==0:
            print('Training example %d/%d' % (j,n_train_batch))
        # grab a random training epoch
        epoch_id=np.random.randint(0,n_valid_ep)
        x_train, y_train=format_ep(raw[:,epoch_id], clean[:,epoch_id], art[:,epoch_id], n_wind, n_tpt, mid_wind)
        train_hist=model_stateful.fit(x_train,
                           y_train,
                           batch_size=batch_size,
                           epochs=1,
                           verbose=0,
                           shuffle=False)
        temp_loss[j]=np.sqrt(train_hist.history['loss'][0])
        model_stateful.reset_states()
    train_loss.append(np.mean(temp_loss))
    train_loss_se.append(stats.sem(temp_loss))

    #Estimate validation error from a few random epochs
    temp_grand_loss=np.zeros(n_valid_batch)
    temp_art_loss = np.zeros(n_valid_batch)
    temp_eeg_loss = np.zeros(n_valid_batch)
    print('Estimating validation error from %d epochs' % n_valid_batch)
    for j in range(n_valid_batch):
        if (j%50)==0:
            print('Validation epoch %d/%d' % (j,n_valid_batch))
        # grab a random validation epoch
        epoch_id = np.random.randint(n_train_ep+n_valid_ep, n_train_ep+n_valid_ep+n_test_ep)
        x, y = format_ep(raw[:, epoch_id], clean[:, epoch_id], art[:, epoch_id], n_wind, n_tpt, mid_wind)
        y_hat = model_stateful.predict(x,batch_size=batch_size)
        temp_eeg_loss[j]= np.sqrt(np.mean(np.square(y[:,0]-y_hat[:,0])))
        temp_art_loss[j]= np.sqrt(np.mean(np.square(y[:,1]-y_hat[:,1])))
        temp_grand_loss[j]= np.sqrt(np.mean(np.square(y - y_hat)))
        model_stateful.reset_states()

    val_eeg_loss.append(np.mean(temp_eeg_loss))
    val_eeg_loss_se.append(stats.sem(temp_eeg_loss))
    val_art_loss.append(np.mean(temp_art_loss))
    val_art_loss_se.append(stats.sem(temp_art_loss))
    val_loss.append(np.mean(temp_grand_loss))
    val_loss_se.append(stats.sem(temp_grand_loss))
    plt.figure(1)
    train_iter = np.arange(1, len(val_loss) + 1)
    plt.clf()
    plt.errorbar(train_iter, np.asarray(val_loss), yerr=np.asarray(val_loss_se)*1.96, fmt='-', label='val')
    plt.errorbar(train_iter, np.asarray(val_art_loss), yerr=np.asarray(val_art_loss_se)*1.96, fmt='-', label='val_art')
    plt.errorbar(train_iter, np.asarray(val_eeg_loss), yerr=np.asarray(val_eeg_loss_se)*1.96, fmt='-', label='val_eeg')
    plt.errorbar(train_iter, np.asarray(train_loss), yerr=np.asarray(train_loss_se) * 1.96, fmt='-', label='train')
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(fig_path, 'epoched_eeg_loss.pdf'))
    np.savez(out_metrics_fname,
             val_eeg_loss=val_eeg_loss,
             val_art_loss=val_art_loss,
             val_loss=val_loss,
             train_loss=train_loss)
    print('Valid error EEG=%f, Art=%f Both=%f (Best=%f)' % (val_eeg_loss[-1],val_art_loss[-1],val_loss[-1],best_val_loss))
    if np.isnan(best_val_loss) or best_val_loss>val_art_loss[-1]:
        # first epoch or improved performance
        best_val_loss=val_art_loss[-1]
        n_last_improvement=0
        print('Best validation loss so far. Saving model as %s' % model_fname)
        model_stateful.save(os.path.join(model_path,model_fname))
    else:
        # No improvment
        n_last_improvement+=1
    if n_last_improvement>=patience:
        print('Validation loss not improving. EXITING!')
        break

# Plot Training & Validation Error
plt.figure(1)
train_iter=np.arange(1,len(val_loss)+1)
plt.clf()
plt.errorbar(train_iter, np.asarray(val_loss), yerr=np.asarray(val_loss_se) * 1.96, fmt='-', label='val')
plt.errorbar(train_iter, np.asarray(val_art_loss), yerr=np.asarray(val_art_loss_se) * 1.96, fmt='-', label='val_art')
plt.errorbar(train_iter, np.asarray(val_eeg_loss), yerr=np.asarray(val_eeg_loss_se) * 1.96, fmt='-', label='val_eeg')
plt.errorbar(train_iter, np.asarray(train_loss), yerr=np.asarray(train_loss_se) * 1.96, fmt='-', label='train')
plt.legend()
# plt.show()
plt.savefig(os.path.join(fig_path, 'epoched_eeg_loss.pdf'))
print('Best valid loss {}'.format(np.min(val_loss)))

print('Loading best model: %s' % model_fname)
model_stateful=load_model(os.path.join(model_path,model_fname))

# Estimate test set error from a few random epochs
temp_art_loss = 0
temp_eeg_loss = 0
print('Estimating test error from %d epochs' % n_test_batch)
for j in range(n_test_batch):
    if (j % 10) == 0:
        print('Testing epoch %d/%d' % (j, n_test_batch))
    # grab a random test epoch
    epoch_id = np.random.randint(n_train_ep+n_valid_ep, n_train_ep + n_valid_ep+n_test_ep)
    x, y = format_ep(raw[:, epoch_id], clean[:, epoch_id], art[:, epoch_id], n_wind, n_tpt, mid_wind)
    y_hat = model_stateful.predict(x, batch_size=batch_size)
    temp_eeg_loss += np.sqrt(np.mean(np.square(y[:, 0] - y_hat[:, 0])))
    temp_art_loss += np.sqrt(np.mean(np.square(y[:, 1] - y_hat[:, 1])))
    model_stateful.reset_states()
    plt.figure(2)
    plt.clf()
    for a in range(2):
        plt.subplot(1,3,a+1)
        plt.plot(y[:,a],'-b',label='y')
        plt.plot(y_hat[:, a],'-r',label='yhat')
        if a==0:
            plt.title('EEG, Epoch %d' % epoch_id)
        else:
            plt.title('Artifact, Epoch %d' % epoch_id)
        plt.legend()
    plt.subplot(1,3,3)
    plt.plot(raw[:,epoch_id],'-b')
    plt.title('Input')
    #plt.show()
    plt.savefig(os.path.join(fig_path,'epoched_eeg_yhat'+str(j)+'.pdf'))
test_eeg_loss=temp_eeg_loss/n_test_batch
test_art_loss=temp_art_loss/n_test_batch
print('** Test Data Loss **')
print('EEG %f' % test_eeg_loss)
print('Art %f' % test_art_loss)
test_loss=(test_eeg_loss+test_art_loss)/2
print('Both %f' % test_loss)

np.savez(out_metrics_fname,
         test_eeg_loss=test_eeg_loss,
         test_art_loss=test_art_loss,
         test_loss=test_loss,
         val_eeg_loss=val_eeg_loss,
         val_art_loss=val_art_loss,
         val_loss=val_loss,
         train_loss=train_loss)
