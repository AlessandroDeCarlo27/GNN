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
from sklearn.model_selection import StratifiedKFold


tf.random.set_seed(12345)
np.random.seed(12345)

#read dataset and drop rows with NaN values
dataset = pd.read_csv('CYP_2C9.csv',sep=",").dropna().reset_index(drop=True)
nameVar = 'CYP_2C9'

dataset,_,_ = filterValidSmiles(dataset)
# Setting dimensions with phantom molecule
setting_nodes_dimension = get_atom_features(rdkit.Chem.MolFromSmiles("CO").GetAtoms()[0])
NFEAT = setting_nodes_dimension.shape[0]

###############################  MODEL DEFINITION  ####################################################################

model = model_definition(NFEAT)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
train_metric = tf.keras.metrics.BinaryCrossentropy(from_logits=True)
val_metric = tf.keras.metrics.AUC(curve='PR')
model.save_weights('ModelStartWeights.hdf5')

############################## KFOLD ####################################################################

# training hyperparameters
idx_fold = 1
NSPLIT = 5
BATCHSIZE = 16
NEPOCHS = 500
BATCH_VERBOSITY = 500

buffer_auc = list()
buffer_auprc = list()

for train_index,test_index in StratifiedKFold(NSPLIT,shuffle=True,random_state=12345).split(dataset,dataset.Y):

    test_metric1 = tf.keras.metrics.AUC(curve='PR')
    test_metric2 = tf.keras.metrics.AUC(curve='ROC')

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

    ######################## weights ########################################################################

    tot = train.shape[0]
    pos = np.sum(train.Y)
    neg = tot-pos
    if pos<neg:
        w_0 = 1
        w_1 = np.log(neg/pos) +1
        weights = [w_0,w_1]
    else:
        weights = [1,1]

    ###################### training ###########################################################################7
    buffer_tloss, buffer_vloss = perform_training_class(model, train, weights, loss, NEPOCHS, BATCHSIZE, optimizer,
                                                  train_metric, validation,
                                                  val_metric, BATCH_VERBOSITY, NFEAT, nameModel, logfile)
    conv_plot_title = "TrainingConvergence" + str(idx_fold) + ".png"
    # training convergence assessment
    plot_training_convergence_class(buffer_tloss, buffer_vloss, NEPOCHS, conv_plot_title)
    recap_df = store_training_info(buffer_tloss, buffer_vloss, NEPOCHS, 'TrainRecap' + str(idx_fold) + '.csv')

    ############################# TEST #############################################################################
    best_model = load_best_model(model, nameModel)

    y_pred_test = perform_test(best_model, test, BATCHSIZE, NFEAT)
    y_real_test = test.Y.to_numpy()
    y_pred_test_prb = tf.nn.sigmoid(y_pred_test)
    auc_test = test_metric2(y_real_test, y_pred_test_prb).numpy()
    auprc_test =  test_metric1(y_real_test, y_pred_test_prb).numpy()

    buffer_auprc.append(auprc_test)
    buffer_auc.append(auc_test)

    plot_ROC(y_real_test, y_pred_test_prb, nameVar, str(idx_fold))
    plot_PRC(y_real_test, y_pred_test_prb, nameVar, str(idx_fold))
    store_recap_test_class(buffer_auprc, buffer_auc, 'testRecap' + str(idx_fold) + '.csv')

    idx_fold+=1
    stop = 1
