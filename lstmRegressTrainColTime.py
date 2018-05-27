""" Simple function for doing multiple linear regression with TensorFlow in Python 3.
 Usage:
  python multLinRegressPy3.py --train trainData.csv

 where trainData.csv is a comma delimited text file. The first column is value
 of the output variable to be predicted. The remaining columns are the input
 predictor variables. Do NOT include a bias predictor. This is taken care of by
 the code.

 Example:
  python multLinRegress.py --train 2dLinRegExample.csv

python lstmRegressTrainColTime.py --train '/Users/davidgroppe/ONGOING/RNN/ECHO_STATE/LORENZ/lorenzData.mat' --name lorenzColTime

python lstmRegressTrainColTime.py --train '/Users/davidgroppe/ONGOING/SZR_STIM/PERSYST_INFO/FULL_DAY_BITEMPORAL/RoNi_79799215711parcChunk1.mat' --name roNiChunk1

trainDataFname='/Users/davidgroppe/PycharmProjects/GROPPE_TF/genSinWave.mat'


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
from datetime import datetime
import argparse

import sys
#sys.path.append('/Users/davidgroppe/PycharmProjects/DG_LIBRARY')
#sys.path.append('/home/h/honey/dgroppe/DG_LIBRARY')
import dgFuncs as dg

startTime = datetime.now()


# PARAMETERS
optimizerType='AdamOptimizer' # Options: SGD, Adagrad, RMSprop
#optimizerType='Adagrad' # Options: SGD, Adagrad, RMSprop
#optimizerType='SGD' # Options: SGD, Adagrad, RMSprop
num_steps = 70001  # originally 7001
summary_frequency = 100
nGen = 80  # Number of time points to generate (80*10 for Lorenz ??)
earlyStoppingPatience = 3  # If validation error fails to go down after this number of time points, quit
regularizationFactor=0.0005 # Set to 0 if you don't want to use this
dropoutProb=0 # I think this is now implemented properly, but it appears to make things worse
sphere=False
validPptn=0.001 # 0.01 for Lorenz
learningRate=0.001 # 0.01 for Lorenz

# HELPER FUNCTIONS
class BatchGenerator(object):
    def __init__(self, dataPts, batch_size, num_unrollings):
        self._dataPts = dataPts
        self._nTpts = dataPts.shape[1]
        self._nDim = dataPts.shape[0]
        self._batch_size = batch_size
        self._num_unrollings = num_unrollings
        segment = self._nTpts // batch_size
        self._cursor = [offset * segment for offset in range(batch_size)]
        self._last_batch = self._next_batch()

    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        batch = np.zeros(shape=(self._batch_size, self._nDim), dtype=np.float)
        for b in range(self._batch_size):
            batch[b, :] = self._dataPts[:, self._cursor[b]]
            self._cursor[b] = (self._cursor[b] + 1) % self._nTpts  # modulus makes sure you don't exceed length of
        return batch

    def next(self):
        """Generate the next array of batches from the data. The array consists of
      the last batch of the previous array, followed by num_unrollings new ones.
      """
        batches = [self._last_batch]
        for step in range(self._num_unrollings):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        return batches


def batches2timeSeries(batches):
    """Convert a sequence of batches back into a time series representation."""
    nUnrollPlus1 = len(batches)
    nBatches = batches[0].shape[0]
    s = []
    for b in range(nBatches):
        s.append(np.zeros((nUnrollPlus1, nDim)))
    for b in range(nBatches):
        for c in range(nUnrollPlus1):
            s[b][c, :] = batches[c][b, :]
    return s


# MAIN BODY
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
    preprocessFname = modelStem + "Preproc"  # File to which data preprocessing parameters (e.g. sphering matrix will be output)

    # Load Data
    print('Loading: {}'.format(trainDataFname))
    matContents = sio.loadmat(trainDataFname)
    data = matContents['data']
    nDim = data.shape[0]
    nTpt = data.shape[1]
    if nDim>nTpt:
        # Transpose data that have been stored using rows for Time
        print('Transposing imported data so that columns represent time')
        del data
        # data = data.T
        data = matContents['data'].T
        nDim = data.shape[0]
        nTpt = data.shape[1]
    del matContents
    print('# of timepoints %d' % nTpt)
    print('# of dimensions %d' % nDim)

    # Create a small validation set.
    # validSize = nTpt//10
    validSize=int(np.round(nTpt*validPptn))
    print('validSize {}'.format(validSize))
    validData=data[:,-validSize:]
    trainData = data[:,:-validSize]
    trainSize = trainData.shape[1]
    print('data.shape')
    print(data.shape)
    del data # delete data to save memory
    print('validData.shape')
    print(validData.shape)
    print('trainData.shape')
    print(trainData.shape)
    print('Training nTpts: {}'.format(trainSize))
    print('Validation nTpts: {}'.format(validSize))

    # Normalize training data
    if False:
        # Check normalization
        print('Pre-normalized data SDs')
        print(trainData.std(axis=1))
        print('Pre-normalized data MNs')
        print(trainData.mean(axis=1))
    if sphere:
        print('Sphering and centering data...')
        trainData, trainDataSph, invTrainDataSph, trainDataMns=dg.sphereCntrData(trainData,epsilon=0)
        validData=dg.applySphereCntr(validData,trainDataSph,trainDataMns)
        print('Saving preprocessing parameters to %s.npz' % preprocessFname)
        np.savez(preprocessFname,trainDataSph=trainDataSph, invTrainDataSph=invTrainDataSph, trainDataMns=trainDataMns)
    else:
        print('Normalizing data to 0 mean, 1 SD...')
        trainData, trainDataMns, trainDataSDs=dg.normalize(trainData)
        validData=dg.applyNormalize(validData,trainDataMns,trainDataSDs)
        print('Saving preprocessing parameters to %s.npz' % preprocessFname)
        np.savez(preprocessFname,trainDataMns=trainDataMns, trainDataSDs=trainDataSDs)
    if False:
        # Check normalization
        print('Post-normalized data SDs')
        print(trainData.std(axis=1))
        print('Post-normalized data MNs')
        print(trainData.mean(axis=1))

    # Plot Beginning the Data
    fig = plt.figure()
    plt.plot(trainData[:,0:200].T, '-o')
    plt.xlabel('Timepoints')
    plt.savefig('PICS/picsLstmReg_initialDataPts.eps')

    # Function to generate a training batch for the LSTM model.
    batch_size = 10  # Number of random contiguous chunks of data to return (10=Ng rule of thumb)
    num_unrollings = 20  # Number of consecutive time points in each chunk

    # Create training and validation batch generators
    train_batches = BatchGenerator(trainData, batch_size, num_unrollings)
    # Each training batch is a length num_unrollings+1 list of batchSize x nDim numpy arrays

    valid_batches = BatchGenerator(validData, 1, 1)
    # Each valid batch is a length 2 list of 1 x nDim numpy arrays

    print('BatchSize {}'.format(batch_size))
    print('NumUnrollings {}'.format(num_unrollings))
    if False:
        # Plot some training batches
        dude = train_batches.next()
        man = batches2timeSeries(dude)
        print('Plotting first two training batches')
        fig = plt.figure()
        showBatch = 0
        plt.plot(man[showBatch][:], '-o')
        plt.title('Training Batch ' + str(showBatch))
        plt.savefig('PICS/picsLstmReg_trainingBatch1.eps')

        fig = plt.figure()
        showBatch = 1
        plt.plot(man[showBatch][:], '-o')
        plt.title('Training Batch ' + str(showBatch))
        plt.savefig('PICS/picsLstmReg_trainingBatch2.eps')

    # Simple LSTM Model.
    num_nodes = 128
    print('num_nodes={}'.format(num_nodes))
    graph = tf.Graph()
    with graph.as_default():
        # Parameters:
        
        #LSTM Cell Weights
        # i=input/write gate, f=forget gate, c=memory gate, o=output, x=input
        ifcox = tf.Variable(tf.truncated_normal([nDim, 4 * num_nodes], mean=0.0, stddev=np.sqrt(2/nDim)))
        ifcom = tf.Variable(tf.truncated_normal([num_nodes, 4 * num_nodes], mean=0.0, stddev=np.sqrt(2/num_nodes)))
        ifcob = tf.Variable(tf.zeros([1, 4*num_nodes]))

        # Variables saving state across unrollings.
        saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
        saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)

        # Classifier weights and biases.
        w = tf.Variable(tf.truncated_normal([num_nodes, nDim], mean=0.0, stddev=np.sqrt(2/num_nodes)))
        b = tf.Variable(tf.zeros([nDim]))

        # Definition of the cell computation.
        def lstm_cell(i, o, state, train=False):
            """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
               Note that in this formulation, we omit the various connections between the
               previous state and the gates."""
            all_gates_state = tf.matmul(i, ifcox) + tf.matmul(o, ifcom) + ifcob
            input_gate = tf.sigmoid(all_gates_state[:, 0:num_nodes])
            forget_gate = tf.sigmoid(all_gates_state[:, num_nodes: 2*num_nodes])
            update = all_gates_state[:, 2*num_nodes: 3*num_nodes]
            state = forget_gate * state + input_gate * tf.tanh(update)
            output_gate = tf.sigmoid(all_gates_state[:, 3*num_nodes:])
            lstmOutput=output_gate * tf.tanh(state)
            if train and dropoutProb>0:
                SEED = 66478  # Set to None for random seed.
                lstmOutput=tf.nn.dropout(lstmOutput, dropoutProb, seed=SEED)
            return lstmOutput, state
        
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
            output, state = lstm_cell(i, output, state, True)
            outputs.append(output)

        # State saving across unrollings.
        with tf.control_dependencies([saved_output.assign(output), saved_state.assign(state)]):
            # Classifier.
            model_outputs = tf.nn.xw_plus_b(tf.concat(0, outputs), w, b)
            # xw_plus_b does matrix multiplication plus bias; not sure what concat is for
            loss = tf.reduce_mean(
                tf.pow(model_outputs - tf.concat(0, train_outputs),2))
            # use mean sqr error for cost function (mean is better than sum as it makes cost independent of batch size)
            if regularizationFactor>0:
                # Add L2 Regularization (biases not included)
                print('Adding L2 regularization with factor %f' % regularizationFactor)
                loss += regularizationFactor * (tf.nn.l2_loss(ifcox) + tf.nn.l2_loss(ifcom) + tf.nn.l2_loss(w))
                # Double check that this is correct??
                # i=input/write gate, f=forget gate, c=memory gate, o=output, x=input
                # w=output classifier wts

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        # Optimizer.
        global_step = tf.Variable(0) # Not sure what this does ??
        if optimizerType=='SGD':
            learning_rate = tf.train.exponential_decay(learningRate, global_step, 5000, 0.1, staircase=True)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        elif optimizerType=='Adagrad':
            # Since Adagrad auto adjusts the learning rate, we shouldn't need to set exponential decay
            learning_rate = tf.Variable(learningRate)
            optimizer = tf.train.AdagradOptimizer(learning_rate)
        elif optimizerType=='RMSProp':
            learning_rate = tf.train.exponential_decay(learningRate, global_step, 5000, 0.1, staircase=True)
            optimizer = tf.train.RMSPropOptimizer(learning_rate)
        elif optimizerType == 'AdamOptimizer':
            learning_rate = tf.Variable(learningRate)
            optimizer = tf.train.AdamOptimizer(learning_rate)
        else:
            print('Error: Unknown optimizer type {}'.format(optimizerType))
            return

        gradients, v = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
        optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

        # Predictions.
        train_prediction = model_outputs # ?? I currently don't use train_prediction for anything

        # Sampling and validation eval: batch 1, no unrolling.
        sample_input = tf.placeholder(tf.float32, shape=[1, nDim])
        saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
        saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
        reset_sample_state = tf.group(saved_sample_output.assign(tf.zeros([1, num_nodes])),
                                      saved_sample_state.assign(tf.zeros([1, num_nodes])))
        sample_output, sample_state = lstm_cell(sample_input, saved_sample_output, saved_sample_state, False)

        with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                      saved_sample_state.assign(sample_state)]):
            sample_prediction = tf.nn.xw_plus_b(sample_output, w, b)

    # TRAINING
    allMnLoss = np.zeros(1 + num_steps // summary_frequency)
    allValidRms = np.zeros(1 + num_steps // summary_frequency)
    stepsSinceLastImprovement = 0
    minValidRms = -1.0  # Running tally of min validation error
    sumCt = 0  # Counts the number of times summary subloop has been run

    validPredictions=np.zeros((nDim,validSize))

    nSubPlotAx1=np.ceil(np.sqrt(nDim))
    plotValidTpts=dg.randTpts(validSize-2,num_unrollings)
    plotGenTpts = dg.randTpts(trainSize, nGen)

    print("nDim={}".format(nDim))
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print('Initialized')
        save_path = saver.save(session, modelFname)
        mean_loss = 0
        for step in range(num_steps):
            batches = train_batches.next()
            feed_dict = dict()
            for i in range(num_unrollings + 1):
                feed_dict[train_data[i]] = batches[i]
            _, l, predictions, lr = session.run(
                    [optimizer, loss, model_outputs, learning_rate], feed_dict=feed_dict)
            mean_loss += l
            if step % summary_frequency == 0:
                if step > 0:
                    mean_loss = mean_loss / summary_frequency
                    # The mean loss is an estimate of the loss over the last few batches.
                allMnLoss[sumCt] = mean_loss
                print('Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
                mean_loss = 0
                if step % (summary_frequency * 10) == 0:
                    # Measure validation set RMS.
                    print('=' * 80)
                    reset_sample_state.run()
                    validRms = 0
                    for validCursor in range(validSize-2):
                        predictions = sample_prediction.eval({sample_input: validData[:,validCursor:validCursor+1].T})  # 1 x nDim np array
                        validRms += np.mean((predictions - validData[:,validCursor+1:validCursor+2].T) ** 2)
                        validPredictions[:,validCursor+1:validCursor+2]=predictions.T #note that we skip the first entry to keep the indexing between validData and validPredictions the same
                    # for _ in range(validSize):
                    #     b = valid_batches.next()
                    #     predictions = sample_prediction.eval({sample_input: b[0]}) # 1 x nDim np array
                    #     validRms += np.mean((predictions - b[1]) ** 2)
                    allValidRms[sumCt] = np.sqrt(validRms / validSize)
                    print('Validation set RMS: %.2f' % allValidRms[sumCt])

                    # Plot predictions and true validation data
                    fig = plt.figure()
                    for sPlotId in range(nDim):
                        plt.subplot(nSubPlotAx1, nSubPlotAx1, sPlotId+1)
                        plt.plot(validData[sPlotId, plotValidTpts].T, 'b-', label='validation')
                        plt.plot(validPredictions[sPlotId, plotValidTpts].T, 'r--', label='predictions')
                        if sPlotId == 0:
                            plt.ylabel('Z-Scores')
                    plt.xlabel('Timing')
                    plt.legend(loc='best')
                    plt.savefig('PICS/picsLstmReg_validPredictions.eps')

                    if (minValidRms > 0):
                        if (minValidRms > allValidRms[sumCt]):
                            minValidRms = allValidRms[sumCt]
                            print('Best model so far.')
                            save_path = saver.save(session, modelFname)
                            print("Model saved in file: %s" % modelFname)
                            # Generate some samples.
                            reset_sample_state.run()
                            for _ in range(1):
                                allGen = np.zeros((nDim,nGen))
                                reset_sample_state.run()
                                # ?? randTpt = np.random.randint(0, trainSize - 1)
                                randTpt=plotGenTpts[0]
                                feed = trainData[:, randTpt:randTpt + 1].T
                                for genCt in range(nGen - 1):
                                    prediction = sample_prediction.eval({sample_input: feed})
                                    allGen[:,genCt + 1] = prediction
                                    feed = prediction

                            # Plot predictions and true validation data
                            fig = plt.figure()
                            for sPlotId in range(nDim):
                                plt.subplot(nSubPlotAx1,nSubPlotAx1,sPlotId+1)
                                plt.plot(trainData[sPlotId, plotGenTpts].T, 'b-', label='training')
                                plt.plot(allGen[sPlotId, :].T, 'r--', label='synthesized')
                                if sPlotId==0:
                                    plt.ylabel('Z-Scores')
                            plt.xlabel('Timing')
                            plt.legend(loc='best')
                            plt.savefig('PICS/picsLstmReg_genVsTrue.eps')

                            stepsSinceLastImprovement=0
                        else:
                            stepsSinceLastImprovement += 1

                        if stepsSinceLastImprovement >= earlyStoppingPatience:
                            print('Validation error not decreasing. Quitting.')
                            break
                    else:
                        minValidRms = allValidRms[sumCt]
                    sumCt += 1
                    print('Steps since last improvement: {}'.format(stepsSinceLastImprovement))

    if stepsSinceLastImprovement < earlyStoppingPatience:
        print('End of requested # of training rounds. Quitting.')

    # Un-Sphere Generated Data
    if sphere:
        allGenRecon = dg.invSphereCntr(allGen, invTrainDataSph, trainDataMns)
    else:
        allGenRecon = allGen

    # Plot generated data
    if nDim==3:
        # 3D Plot of generated data
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(allGenRecon[0, 0:nGen], allGenRecon[1, 0:nGen], allGenRecon[2, 0:nGen], '-')
    else:
        # Butterfly plot of generated data
        fig = plt.figure()
        plt.plot(allGenRecon.T, '-o')

    plt.title('Synthesized Data')
    plt.savefig('PICS/picsLstmReg_generatedData.eps')

    # Display some parameters
    print('# of nodes: {}'.format(num_nodes))
    print('learningRate: {}'.format(learningRate))
    print('Optimizer: {}'.format(optimizerType))

    # Plot training and validation data
    print('Validation Error:')
    print(allValidRms[allValidRms > 0])
    fig = plt.figure()
    plt.plot(allMnLoss[allValidRms > 0], label='training')
    plt.plot(allValidRms[allValidRms > 0], 'r-', label='validation')
    plt.xlabel('Training Step/' + str(summary_frequency))
    plt.ylabel('Mean Loss')
    plt.legend(loc='best')
    plt.savefig('PICS/picsLstmReg_trainingError.eps')

    # Output elapsed time
    print(datetime.now() - startTime)

if __name__ == '__main__':
    tf.app.run()
