nameModelBase = 'MHATT_bonds_deeper_benchmark_clipping_weights'
logfile = open('Log_MHATT_bonds_deeper_benchmark_clipping_weights.txt','w')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import pandas as pd
import rdkit, rdkit.Chem, rdkit.Chem.rdDepictor, rdkit.Chem.Draw
import networkx as nx
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.utils import gen_batches
import time
from utils.GNNlayers import *
from utils.ProcessData import *
from utils.TrainModel import *
from utils.TestModel import *
from sklearn.model_selection import KFold


tf.random.set_seed(12345)
np.random.seed(12345)

#read dataset and drop rows with NaN values
dataset = pd.read_csv('Lipophilicity.csv',sep=",").dropna().reset_index(drop=True)

dataset,_,_ = filterValidSmiles(dataset)
# Setting dimensions with phantom molecule
setting_nodes_dimension = get_atom_features(rdkit.Chem.MolFromSmiles("CO").GetAtoms()[0])
NFEAT = setting_nodes_dimension.shape[0]

###############################  MODEL DEFINITION  ####################################################################

model = model_definition(NFEAT)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

train_metric = tf.keras.metrics.RootMeanSquaredError()
val_metric = tf.keras.metrics.RootMeanSquaredError()
model.save_weights('ModelStartWeights.hdf5')
############################## KFOLD ####################################################################

nameVar = 'LogD'
# training hyperparameters
idx_fold = 1
NSPLIT = 5
BATCHSIZE = 16
NEPOCHS = 500
BATCH_VERBOSITY = 500


# results buffers
buffer_rmse = list()
buffer_mae = list()
buffer_r2 = list()

for train_index,test_index in KFold(NSPLIT,shuffle=True,random_state=12345).split(dataset):
    # initialize model weights
    model.load_weights('ModelStartWeights.hdf5')
    # update model name
    nameModel = nameModelBase + '_F'+str(idx_fold)
    print("-----******** K FOLD # "+str(idx_fold)+" ********------",file=logfile)

    ####################### DATA PREPARATION ################################################################

    TV = dataset[["SMILES", "Y"]].iloc[train_index].sample(frac=1, random_state=12345).reset_index(drop=True)
    test = dataset[["SMILES","Y"]].iloc[test_index].sample(frac=1,random_state=12345).reset_index(drop=True)
    train, validation = train_test_split(TV, test_size=0.2, shuffle=True, random_state=12345)
    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)

    #weighting training example

    # build the knernel density estimator
    kde = KernelDensity(kernel="gaussian", bandwidth=0.2).fit(train.Y.to_numpy()[:, np.newaxis])
    # evaluate the goodness of the kernel estimator
    ex_scores = np.exp(kde.score_samples(np.linspace(-2, 5, 500)[:, np.newaxis]))
    # create object WeightTrain with the given kernel
    weight_generator = WeightTrain(train.Y.to_numpy(), kde, 0.55)
    scores_train = weight_generator.getWeigths(np.linspace(-2, 5, 500)[:, np.newaxis])
    range = (-2,5)
    ## plot distributions and weights of the training examples ##
    plot_weights(train, scores_train, ex_scores, idx_fold,nameVar,range,500)

    ############################## TRAINING ######################################################################
    buffer_tloss, buffer_vloss = perform_training(model, train, weight_generator, NEPOCHS, BATCHSIZE, optimizer,
                                                  train_metric, validation,
                                                  val_metric, BATCH_VERBOSITY, NFEAT, nameModel, logfile)

    conv_plot_title = "TrainingConvergence" + str(idx_fold) + ".png"
    # training convergence assessment
    plot_training_convergence(buffer_tloss, buffer_vloss, NEPOCHS, conv_plot_title)
    recap_df = store_training_info(buffer_tloss, buffer_vloss, NEPOCHS, 'TrainRecap' + str(idx_fold) + '.csv')

    ############################# TEST #############################################################################
    best_model = load_best_model(model, nameModel)

    # compute statistics
    y_pred_test = perform_test(best_model, test, BATCHSIZE, NFEAT)
    y_real_test = test.Y.to_numpy()
    res = y_real_test - y_pred_test
    rmse_test = tf.sqrt(tf.reduce_mean(tf.square((y_pred_test - y_real_test)))).numpy()
    mae_test = tf.reduce_mean(tf.abs((y_pred_test - y_real_test))).numpy()
    r2 = (1 - (tf.reduce_sum(tf.square((y_pred_test - y_real_test))) / tf.reduce_sum(
        tf.square(y_real_test - tf.reduce_mean(y_real_test))))).numpy()

    buffer_mae.append(mae_test)
    buffer_rmse.append(rmse_test)
    buffer_r2.append(r2)


    # residuals
    scatterplot(y_real_test, y_pred_test, nameVar, idx_fold)
    residuals(y_real_test, res, nameVar, idx_fold)
    store_recap_test(buffer_rmse, buffer_mae, buffer_r2, 'testRecap' + str(idx_fold) + '.csv')



    idx_fold +=1
