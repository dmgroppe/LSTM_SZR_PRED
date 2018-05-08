""" Simple function for doing multiple linear regression with TensorFlow in Python 3.
 Usage:
  python multLinRegressPy3.py --train trainData.csv

 where trainData.csv is a comma delimited text file. The first column is value
 of the output variable to be predicted. The remaining columns are the input
 predictor variables. Do NOT include a bias predictor. This is taken care of by
 the code.

 Example:
  python multLinRegress.py --train 2dLinRegExample.csv

 Code based on Jason Baldrige's softmax.py function:
  https://github.com/jasonbaldridge/try-tf

trainDataFname='/Users/davidgroppe/PycharmProjects/GROPPE_TF/genSinWave.mat'
/Users/davidgroppe/ONGOING/RNN/ECHO_STATE/LORENZ/lorenzData.mat

python lstmRegressGenColTime.py --train /Users/davidgroppe/ONGOING/RNN/ECHO_STATE/LORENZ/lorenzData.mat

 David Groppe
 Python newbie
 Dec, 2015

"""

# ?? some of these I could probably remove?
from __future__ import print_function
import os
import numpy as np
import random
import tensorflow as tf
import zipfile
from six.moves import range
import matplotlib.pyplot as plt
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
import argparse

import sys
sys.path.append('/Users/davidgroppe/PycharmProjects/DG_LIBRARY')
import dgFuncs as dg


# Define the flags useable from the command line.
tf.app.flags.DEFINE_string('train', None,
                           'File containing the training data (labels & features).')
FLAGS = tf.app.flags.FLAGS

batch_size = 10  # Number of random contiguous chunks of data to return (10=Ng rule of thumb)
num_unrollings = 20  # Number of consecutive time points in each chunk
nGen = 80*100  # Number of time points to generate


def main(argv=None):
    # Read command line arguments
    parser = argparse.ArgumentParser(description='Trains an LSTM on time series data.')
    parser.add_argument('-t', '--train', help='Training data file name', required=True)
    parser.add_argument('-n', '--name', help='Model name (used for output files)', required=True)
    args = parser.parse_args() # of type <class 'argparse.Namespace'>

    trainDataFname = args.train
    print("Training Data: {}".format(trainDataFname))
    modelStem = args.name
    modelFname = modelStem + ".ckpt"
    preprocessFname = modelStem + "Preproc.npz"  # File to which data preprocessing parameters (e.g. sphering matrix will be output)

    # Load Data
    print('Loading data for generation initialization/comparison: {}'.format(trainDataFname))
    matContents = sio.loadmat(trainDataFname)
    data = matContents['data']
    nDim = data.shape[0]
    nTpt = data.shape[1]
    if nDim>nTpt:
        # Transpose data that have been stored using rows for Time
        print('Transposing imported data so that columns represent time')
        del data
        data = matContents['data'].T
        nDim = data.shape[0]
        nTpt = data.shape[1]
    print('# of timepoints %d' % nTpt)
    print('# of dimensions %d' % nDim)
    del matContents

    # Load Preprocessing Parameters
    preproc=np.load(preprocessFname)
    sphere = 'trainDataSph' in preproc.keys()
    if sphere:
        print('Sphering and centering data...')
        # need load values from training ??
        #data, dataSph, dataSph, dataMns=dg.sphereCntrData(data,epsilon=0)
        data = dg.applySphereCntr(data, preproc['trainDataSph'], preproc['trainDataMns'])
    else:
        print('Normalizing data to 0 mean, 1 SD...')
        #data, dataMns, dataSDs=dg.normalize(data)
        data=dg.applyNormalize(data, preproc['trainDataMns'], preproc['trainDataSDs'])


    # Simple LSTM Model.
    num_nodes = 64
    graph = tf.Graph()
    with graph.as_default():
        # Parameters:

        #LSTM Cell Weights
        # i=input/write, f=forget, c=memory, o=output
        ifcox = tf.Variable(tf.truncated_normal([nDim, 4 * num_nodes], -0.1, 0.1))
        ifcom = tf.Variable(tf.truncated_normal([num_nodes, 4 * num_nodes], -0.1, 0.1))
        ifcob = tf.Variable(tf.zeros([1, 4*num_nodes]))

        # Variables saving state across unrollings.
        saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
        saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
        # Classifier weights and biases.
        w = tf.Variable(tf.truncated_normal([num_nodes, nDim], -0.1, 0.1))
        b = tf.Variable(tf.zeros([nDim]))

        # Definition of the cell computation.
        def lstm_cell(i, o, state):
            """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
            Note that in this formulation, we omit the various connections between the
                previous state and the gates."""
            all_gates_state = tf.matmul(i, ifcox) + tf.matmul(o, ifcom) + ifcob
            input_gate = tf.sigmoid(all_gates_state[:, 0:num_nodes])
            forget_gate = tf.sigmoid(all_gates_state[:, num_nodes: 2*num_nodes])
            update = all_gates_state[:, 2*num_nodes: 3*num_nodes]
            state = forget_gate * state + input_gate * tf.tanh(update)
            output_gate = tf.sigmoid(all_gates_state[:, 3*num_nodes:])
            return output_gate * tf.tanh(state), state

        # Input data.
        train_data = list()
        for _ in range(num_unrollings + 1):
            train_data.append(
                tf.placeholder(tf.float32, shape=[batch_size, nDim]))
        train_inputs = train_data[:num_unrollings]
        train_outputs = train_data[1:]  # labels are inputs shifted by one time step.

        # Unrolled LSTM loop.
        outputs = list()
        output = saved_output
        state = saved_state
        for i in train_inputs:
            output, state = lstm_cell(i, output, state)
            outputs.append(output)

        # State saving across unrollings.
        with tf.control_dependencies([saved_output.assign(output), saved_state.assign(state)]):
            # Classifier.
            model_outputs = tf.nn.xw_plus_b(tf.concat(0, outputs), w, b) # xw_plus_b does matrix multiplication plus bias; not sure what concat is for
            loss = tf.reduce_mean(
                tf.pow(model_outputs - tf.concat(0, train_outputs),2)) # use mean sqr error for cost function (mean is better than sum as it makes cost independent of batch size)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        # Optimizer.
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(0.1, global_step, 5000, 0.1, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        #optimizer = tf.train.AdagradOptimizer(learning_rate)
        #optimizer = tf.train.RMSPropOptimizer(learning_rate)

        gradients, v = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
        optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

        # Predictions.
        train_prediction = model_outputs

        # Sampling and validation eval: batch 1, no unrolling.
        sample_input = tf.placeholder(tf.float32, shape=[1, nDim])
        saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
        saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
        reset_sample_state = tf.group(saved_sample_output.assign(tf.zeros([1, num_nodes])),
                                      saved_sample_state.assign(tf.zeros([1, num_nodes])))
        sample_output, sample_state = lstm_cell(sample_input, saved_sample_output, saved_sample_state)

        with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                      saved_sample_state.assign(sample_state)]):
            sample_prediction = tf.nn.xw_plus_b(sample_output, w, b)


    print("nDim={}".format(nDim))

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print('Initialized')
        print('Loading model from %s' % modelFname)
        saver.restore(session,modelFname)

        # Generate some samples.
        reset_sample_state.run()
        for _ in range(1):
            #print('First input is a random number.')
            #feed = np.random.randn(1, nDim)
            print('First input is a random training data time point.')
            randTpt=np.random.randint(0,nTpt-1)
            feed = data[:,randTpt:randTpt+1].T
            allGen = np.zeros((nDim, nGen))
            reset_sample_state.run()
            for genCt in range(nGen - 1):
                prediction = sample_prediction.eval({sample_input: feed})
                allGen[:,genCt + 1] = prediction # ?? might have to transpose this
                feed = prediction

    if sphere:
        # Un-Sphere Data
        #allGenRecon = dg.invSphereCntr(allGen, invTrainDataSph, trainDataMns)
        allGenRecon = dg.invSphereCntr(allGen, preproc['invTrainDataSph'], preproc['trainDataMns'])
        data = dg.invSphereCntr(data, preproc['invTrainDataSph'], preproc['trainDataMns'])
    else:
        allGenRecon = allGen

    if nDim==3:
        # Plot generated and training data in 3D
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(allGenRecon[0,0:nGen], allGenRecon[1,0:nGen], allGenRecon[2,0:nGen], '-')
        plt.title('Synthesized Data')
        plt.savefig('PICS/GENpicsLstmReg_generatedData.eps')

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(data[0,0:nGen], data[1,0:nGen], data[2,0:nGen], '-')
        plt.title('Training Data')
        plt.savefig('PICS/GENpicsLstmReg_trainingData.eps')
    else:
        # Plot generated and training data as butterfly plots
        fig = plt.figure()
        plt.plot(allGenRecon.T, '-o')
        plt.title('Synthesized Data')
        plt.savefig('PICS/GENpicsLstmReg_generatedData.eps')

        fig = plt.figure()
        plt.plot(data.T, '-o')
        plt.title('Training Data')
        plt.savefig('PICS/GENpicsLstmReg_trainingData.eps')

    print('Done generating data!')

if __name__ == '__main__':
    tf.app.run()
