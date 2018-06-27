""" Word up. """
from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import sys
from keras.models import Sequential, load_model
from keras.layers import Dense, SimpleRNN, Activation, Dropout, Merge
if sys.platform=='darwin':
    from keras.layers import GRU
else:
    from keras.layers import CuDNNGRU
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

def inv_normalize(data, norm_fact):
    """ data - data matrix
        norm_fact - tuple of floats; norm_fact[0]=centering factor; norm_fact[1]=shrinkage factor"""
    return (data*norm_fact[1])+norm_fact[0]


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
    print('Usage: test_eeg_gru_relu_epoched.py model_name')
    exit()
if len(sys.argv)!=2:
    raise Exception('Error, code requires 1 argument: model_name')

# Import Parameters from json file
model_name=sys.argv[1]
print('Importing model %s' % model_name)

# Find if there are any existing models of this name
# If so, grab the number of model an increment that number by 1 to get new model name
if sys.platform=='darwin':
    model_root = '/Users/davidgroppe/PycharmProjects/LSTM_SZR_PRED/MODELS/'
else:
    model_root='/home/dgroppe/GIT/LSTM_SZR_PRED/MODELS/'
model_path=os.path.join(model_root,model_name)
print('Model will be stored to %s' % model_path)
fig_path = os.path.join(model_path, 'PICS')

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

print('Loading Model...')
model_fname=os.path.join(model_path,'cuda_gru_epoched.h5')
model=load_model(os.path.join(model_path,model_fname))
model.summary()

batch_size = 1
n_wind=31
mid_wind=int(np.ceil(n_wind/2))
subsample=False
if subsample==True:
    # Estimate test set error from a few random epochs
    n_test_batch=300 # # of test epochs to plot and estimate testing error on
    test_art_loss = np.zeros(n_test_batch)
    test_eeg_loss = np.zeros(n_test_batch)
    test_both_loss=np.zeros(n_test_batch)
    print('Estimating test error from %d epochs' % n_test_batch)
    for j in range(n_test_batch):
        if (j % 10) == 0:
            print('Testing epoch %d/%d' % (j, n_test_batch))
        # grab a random test epoch
        epoch_id = np.random.randint(n_train_ep+n_valid_ep, n_train_ep + n_valid_ep+n_test_ep)
        x, y = format_ep(raw[:, epoch_id], clean[:, epoch_id], art[:, epoch_id], n_wind, n_tpt, mid_wind)
        y_hat = model.predict(x, batch_size=batch_size)
        test_eeg_loss[j] = np.sqrt(np.mean(np.square(y[:, 0] - y_hat[:, 0])))
        test_art_loss[j] = np.sqrt(np.mean(np.square(y[:, 1] - y_hat[:, 1])))
        test_both_loss[j] = np.sqrt(np.mean(np.square(y - y_hat)))
        model.reset_states()
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
    # test_eeg_loss=test_eeg_loss/n_test_batch
    # test_art_loss=test_art_loss/n_test_batch
    print('** Test Data Loss **')
    print('EEG %f (+-%f)' % (np.mean(test_eeg_loss),stats.sem(test_eeg_loss)))
    print('Art %f (+-%f)' % (np.mean(test_art_loss),stats.sem(test_art_loss)))
    test_loss=(test_eeg_loss+test_art_loss)/2
    print('Both %f (+-%f)' % (np.mean(test_both_loss),stats.sem(test_both_loss)))
else:
    # Apply filter to all test data
    print('Applying model to all test epochs')
    art_yhat=np.zeros((n_tpt-n_wind,n_test_ep))
    eeg_yhat = np.zeros((n_tpt-n_wind, n_test_ep))
    art_y=np.zeros((n_tpt-n_wind,n_test_ep))
    eeg_y = np.zeros((n_tpt-n_wind, n_test_ep))
    for ct, epoch_id in enumerate(test_ids):
        if (ct % 10) == 0:
            print('Test epoch %d/%d' % (ct, n_test_ep))
        x, y = format_ep(raw[:, epoch_id], clean[:, epoch_id], art[:, epoch_id], n_wind, n_tpt, mid_wind)
        y_hat = model.predict(x, batch_size=batch_size)
        eeg_yhat[:,ct]=y_hat[:,0]
        eeg_y[:, ct] = y[:,0]
        art_yhat[:,ct]=y_hat[:,1]
        art_y[:, ct] = y[:,1]
        model.reset_states()
    # rescale filter output to match input
    eeg_y=inv_normalize(eeg_y,clean_nrm_fact)
    eeg_yhat = inv_normalize(eeg_yhat, clean_nrm_fact)
    art_y=inv_normalize(art_y,art_nrm_fact)
    art_yhat = inv_normalize(art_yhat, art_nrm_fact)
    #clean_nrm_fact, art_nrm_fact

    out_data_fname='filtered_test_data.npz'
    print('Saving test filter output to %s' % out_data_fname)
    np.savez(out_data_fname,
             model_name=model_name,
             test_ids=test_ids,
             eeg_yhat=eeg_yhat,
             eeg_y=eeg_y,
             art_yhat=art_yhat,
             art_y=art_y,
             n_wind=n_wind)
