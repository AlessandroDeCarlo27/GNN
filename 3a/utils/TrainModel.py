import tensorflow as tf
import numpy as np
from utils.ProcessData import get_maxDims, gen_smiles2graph
from sklearn.neighbors import KernelDensity
from sklearn.utils import gen_batches
from utils.TestModel import perform_validation, perform_validationPPB, perform_validationCL, perform_validation_class, perform_validation_class_mt
from sklearn.utils import gen_batches
import pandas as pd
import time
import matplotlib.pyplot as plt

def loss(y_true, y_pred,w):
    """
    Weighted RMSE
    """
    return tf.keras.backend.sqrt(tf.keras.backend.mean(w*tf.keras.backend.square(y_pred - y_true)))

class WeightTrain:
    def __init__(self,Y,kde,alpha,epsilon=1e-6):
        """
        Y: values of dependent variable in traning set. Il must be np.array
        kde: kde object of sklearn library
        alpha: hyperparameters of the algorithm. alpha=0, no weight; higher values of alpha increase the discrepancy
               between the weights of the extreme values and the most typical
        epsilon: default value when the weight is too small
        """
        self.alpha = alpha
        self.epsilon = epsilon
        self.kde = kde

        if Y.ndim == 1:
            Y = Y[:,np.newaxis]

        y_probs = np.exp(self.kde.score_samples(Y))

        self.MAX = y_probs.max()
        self.MIN = y_probs.min()
        self.DEN = self.MAX-self.MIN
    def getWeigths(self,X):
        norm_score = (np.exp(self.kde.score_samples(X))-self.MIN)/self.DEN
        out_score = 1-self.alpha*norm_score
        out_score[out_score<self.epsilon] = self.epsilon
        return out_score


def eval_saving_wt(model,nameModel,buffer_vloss):

    """
    Function that evaluates if model weights should be saved according to the latest value of the loss on the validation
    INPUT:
    - model: object representing the model
    - nameModel: string with the name of the model
    - buffer_vloss: list with the loss values computed in the epochs
    """


    performed_ep = len(buffer_vloss)
    if performed_ep == 0:
        return None
    elif performed_ep == 1:
        model.save_weights(nameModel+'.hdf5')
        return 'val loss improved from inf to '+str(buffer_vloss[-1])+' -- saving weights to '+nameModel+'.hdf5'
    else:
        if buffer_vloss[-1]<np.array(buffer_vloss)[0:performed_ep-1].min():
            model.save_weights(nameModel + '.hdf5')
            return 'val loss improved from '+ str(np.array(buffer_vloss)[0:performed_ep-1].min()) \
                   + ' to '+ str(buffer_vloss[-1]) +\
                   ' -- saving weights to '+nameModel+'.hdf5'
        else:
            return 'val loss did not improve'

def perform_training(model,train,weight_generator,NEPOCHS,BATCHSIZE,optimizer,train_metric,validation,
                   val_metric,BATCH_VERBOSITY,NFEAT,nameModel,logfile):
    buffer_tloss = list()
    buffer_vloss = list()
    for epoch in range(NEPOCHS):
        tStart = time.time()
        print('----- EPOCH', str(epoch + 1), '/', str(NEPOCHS), '-----', file=logfile)
        # shuffling
        train = train.sample(frac=1, random_state=12345).reset_index(drop=True)
        # gen batches
        batch_train = gen_batches(len(train), BATCHSIZE)
        for idx, b in enumerate(batch_train):
            bt = train[b]
            feat_list = list()
            nm_list1 = list()
            nm_list2 = list()
            nm_list3 = list()
            nm_list4 = list()
            cm_list1 = list()
            cm_list2 = list()
            cm_list3 = list()
            cm_list4 = list()
            mask_list = list()
            mask_list1 = list()
            mask_list2 = list()
            mask_list3 = list()
            mask_list4 = list()
            nmall_list = list()
            cmall_list = list()
            y_real = bt.Y.to_numpy()
            if BATCHSIZE > 1:
                maxDim_nodes, maxDim_links = get_maxDims(bt, NFEAT)
            for smile in bt.SMILES:
                feat, NmatList, CmatList, mask, mask_list_mol,Nall_mol,Call_mol = gen_smiles2graph(smile, NFEAT)

                if BATCHSIZE > 1:  # DO PADDING

                    new_feat = np.zeros((maxDim_nodes, feat.shape[1]))  # dimensione spazio latente
                    new_feat[:feat.shape[0], :feat.shape[1]] = feat



                    for k in range(0, len(NmatList)):
                        new_Nmat = np.zeros((maxDim_links, maxDim_nodes))
                        new_Nmat[:NmatList[k].shape[0], :NmatList[k].shape[1]] = NmatList[k]
                        NmatList[k] = new_Nmat

                    for k in range(0, len(CmatList)):
                        new_Cmat = np.zeros((maxDim_links, maxDim_nodes))
                        new_Cmat[:CmatList[k].shape[0], :CmatList[k].shape[1]] = CmatList[k]
                        CmatList[k] = new_Cmat

                    for k in range(0,len(mask_list_mol)):
                        new_mask = np.zeros((1, maxDim_nodes))
                        new_mask[0, :feat.shape[0]] = mask_list_mol[k]
                        mask_list_mol[k] = new_mask

                    new_Nmat_all = np.zeros((maxDim_links, maxDim_nodes))
                    new_Nmat_all[:Nall_mol.shape[0], :Nall_mol.shape[1]] = Nall_mol

                    new_Cmat_all = np.zeros((maxDim_links, maxDim_nodes))
                    new_Cmat_all[:Call_mol.shape[0], :Call_mol.shape[1]] = Call_mol

                    new_mask = np.zeros((1, maxDim_nodes))
                    new_mask[0, :feat.shape[0]] = mask

                else:  # NO PADDING
                    new_feat = feat
                    new_mask = mask
                    new_Nmat_all = Nall_mol
                    new_Cmat_all = Call_mol
                nm1, nm2, nm3, nm4 = NmatList
                cm1, cm2, cm3, cm4 = CmatList
                mk1,mk2,mk3,mk4 = mask_list_mol

                feat_list.append(new_feat)

                nm_list1.append(nm1)
                nm_list2.append(nm2)
                nm_list3.append(nm3)
                nm_list4.append(nm4)

                cm_list1.append(cm1)
                cm_list2.append(cm2)
                cm_list3.append(cm3)
                cm_list4.append(cm4)

                mask_list.append(new_mask)

                mask_list1.append(mk1)
                mask_list2.append(mk2)
                mask_list3.append(mk3)
                mask_list4.append(mk4)

                nmall_list.append(new_Nmat_all)
                cmall_list.append(new_Cmat_all)

            feat_batch = np.stack(feat_list)

            nm1_batch = np.stack(nm_list1)
            nm2_batch = np.stack(nm_list2)
            nm3_batch = np.stack(nm_list3)
            nm4_batch = np.stack(nm_list4)

            cm1_batch = np.stack(cm_list1)
            cm2_batch = np.stack(cm_list2)
            cm3_batch = np.stack(cm_list3)
            cm4_batch = np.stack(cm_list4)

            mask_batch = np.stack(mask_list)

            mask_batch1 = np.stack(mask_list1)
            mask_batch2 = np.stack(mask_list2)
            mask_batch3 = np.stack(mask_list3)
            mask_batch4 = np.stack(mask_list4)

            nmall_batch = np.stack(nmall_list)
            cmall_batch = np.stack(cmall_list)

            # registro le operazioni runnate durante il forward
            # in questo modo consento l'auto-differentiation
            with tf.GradientTape() as tape:
                # forward pass
                y_pred = tf.squeeze(model((feat_batch, nm1_batch, cm1_batch,
                                           nm2_batch, cm2_batch,
                                           nm3_batch, cm3_batch,
                                           nm4_batch, cm4_batch,
                                           mask_batch1, mask_batch2,
                                           mask_batch3, mask_batch4,
                                           nmall_batch, cmall_batch,
                                           mask_batch), training=True))
                # loss
                we = weight_generator.getWeigths(y_real[:, np.newaxis])
                loss_value = loss(y_real, y_pred, we)
            # gradiente parametri rispetto la loss
            grads = tape.gradient(loss_value, model.trainable_weights)
            grads, _ = tf.clip_by_global_norm(grads, 5.0)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            if y_pred.numpy().ndim == 0:
                y_pred = tf.reshape(y_pred, shape=(1,))
            train_metric.update_state(y_real, y_pred,sample_weight=we)
            if (idx + 1) % BATCH_VERBOSITY == 0:
                train_metric_printable = train_metric.result()
                print('Epoch', str(epoch + 1), '- after', str(idx + 1), 'batches training loss is:',
                      str(float(train_metric_printable)), file=logfile)
        tEnd = time.time()
        train_metric_printable = train_metric.result()
        buffer_tloss.append(train_metric.result().numpy())
        train_metric.reset_states()
        val_loss = perform_validation(model, validation, val_metric, batchsize=BATCHSIZE, nfeat=NFEAT)
        buffer_vloss.append(val_loss)
        print('End of Epoch', str(epoch + 1), '- time:', str(round(tEnd - tStart, 2)), '- training loss:',
              str(float(train_metric_printable)), '- val loss: ', str(float(val_loss)), file=logfile)
        out_msg = eval_saving_wt(model, nameModel, buffer_vloss)
        if not (out_msg is None):
            print('Epoch ', str(epoch + 1), ': ', out_msg, file=logfile)
    return buffer_tloss, buffer_vloss


def plot_training_convergence(buffer_tloss,buffer_vloss,NEPOCHS,title):
    best_epoch = np.array(buffer_vloss).argmin()
    best_val = np.array(buffer_vloss)[best_epoch]
    best_epoch += 1
    allv = np.concatenate([buffer_vloss, buffer_tloss])
    sup_lim = allv.max()
    inf_lim = allv.min() * 0.95

    plt.rcParams.update({'font.size': 36})
    plt.plot(np.arange(1, NEPOCHS + 1), np.array(buffer_tloss), linewidth=5)
    plt.plot(np.arange(1, NEPOCHS + 1), np.array(buffer_vloss), linewidth=5)
    plt.vlines(best_epoch, inf_lim, sup_lim, linestyles='dotted', color='black')
    plt.plot(best_epoch, best_val, 'og', label='Best', markersize=15)
    plt.legend(['Training [WRMSE]', 'Validation [RMSE]', 'Best'], framealpha=1)
    plt.grid()
    plt.title('Training Convergence')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    fig = plt.gcf()
    fig.set_size_inches(32, 18)
    plt.savefig(title, bbox_inches='tight', dpi=250)
    plt.clf()

def plot_training_convergence_class(buffer_tloss,buffer_vloss,NEPOCHS,title):
    best_epoch = np.array(buffer_vloss).argmax()
    best_val = np.array(buffer_vloss)[best_epoch]
    best_epoch += 1
    allv = np.concatenate([buffer_vloss, buffer_tloss])
    sup_lim = allv.max()
    inf_lim = allv.min() * 0.95

    plt.rcParams.update({'font.size': 36})
    plt.plot(np.arange(1, NEPOCHS + 1), np.array(buffer_tloss), linewidth=5)
    plt.plot(np.arange(1, NEPOCHS + 1), np.array(buffer_vloss), linewidth=5)
    plt.vlines(best_epoch, inf_lim, sup_lim, linestyles='dotted', color='black')
    plt.plot(best_epoch, best_val, 'og', label='Best', markersize=15)
    plt.legend(['Training [WBCE]', 'Validation [AUPRC]', 'Best'], framealpha=1)
    plt.grid()
    plt.title('Training Convergence')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    fig = plt.gcf()
    fig.set_size_inches(32, 18)
    plt.savefig(title, bbox_inches='tight', dpi=250)
    plt.clf()

def store_training_info(buffer_tloss,buffer_vloss,NEPOCHS,filename):

    ep_list = np.arange(1,NEPOCHS+1,1).tolist()
    recap_df = pd.DataFrame(list(zip(ep_list, buffer_tloss, buffer_vloss)),
                            columns=['Epoch', 'TrainingLoss', 'ValLoss'])

    recap_df.to_csv(filename,index=False)
    return recap_df


def plot_weights(train,scores_train,ex_scores,idx_fold,nameVar,range,Npoints):
    plt.rcParams.update({'font.size': 36})
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    fig.set_size_inches(32, 18)
    ax1.set_facecolor('#F5F5F5')
    ax1.grid()
    ax1.hist(train.Y, density=True)
    ax1.plot(np.linspace(range[0], range[1], Npoints)[:, np.newaxis], ex_scores, linewidth=5)
    ax1.legend(["Training hist", "KDE"])
    ax1.set_title("Distribution of training "+nameVar)
    ax1.set_ylabel("pdf")
    ax2.plot(np.linspace(range[0], range[1], Npoints)[:, np.newaxis], scores_train, 'g', linewidth=5)
    ax2.set_facecolor('#F5F5F5')
    ax2.grid()
    ax2.set_title("Distribution of training weights")
    ax2.set_ylabel("Weight")
    ax2.set_xlabel(nameVar)
    ax2.set_ylim(0,1.1)
    plt.savefig('ScoreAndDensity' + str(idx_fold) + '.jpg', bbox_inches='tight', dpi=250)
    plt.clf()


def perform_trainingPPB(model,train,weight_generator,NEPOCHS,BATCHSIZE,optimizer,train_metric,validation,
                   val_metric,BATCH_VERBOSITY,NFEAT,nameModel,logfile):
    buffer_tloss = list()
    buffer_vloss = list()
    for epoch in range(NEPOCHS):
        tStart = time.time()
        print('----- EPOCH', str(epoch + 1), '/', str(NEPOCHS), '-----', file=logfile)
        # shuffling
        train = train.sample(frac=1, random_state=12345).reset_index(drop=True)
        # gen batches
        batch_train = gen_batches(len(train), BATCHSIZE)
        for idx, b in enumerate(batch_train):
            bt = train[b]
            feat_list = list()
            nm_list1 = list()
            nm_list2 = list()
            nm_list3 = list()
            nm_list4 = list()
            cm_list1 = list()
            cm_list2 = list()
            cm_list3 = list()
            cm_list4 = list()
            mask_list = list()
            mask_list1 = list()
            mask_list2 = list()
            mask_list3 = list()
            mask_list4 = list()
            nmall_list = list()
            cmall_list = list()
            y_real = bt.Y.to_numpy()
            if BATCHSIZE > 1:
                maxDim_nodes, maxDim_links = get_maxDims(bt, NFEAT)
            for smile in bt.SMILES:
                feat, NmatList, CmatList, mask, mask_list_mol,Nall_mol,Call_mol = gen_smiles2graph(smile, NFEAT)

                if BATCHSIZE > 1:  # DO PADDING

                    new_feat = np.zeros((maxDim_nodes, feat.shape[1]))  # dimensione spazio latente
                    new_feat[:feat.shape[0], :feat.shape[1]] = feat



                    for k in range(0, len(NmatList)):
                        new_Nmat = np.zeros((maxDim_links, maxDim_nodes))
                        new_Nmat[:NmatList[k].shape[0], :NmatList[k].shape[1]] = NmatList[k]
                        NmatList[k] = new_Nmat

                    for k in range(0, len(CmatList)):
                        new_Cmat = np.zeros((maxDim_links, maxDim_nodes))
                        new_Cmat[:CmatList[k].shape[0], :CmatList[k].shape[1]] = CmatList[k]
                        CmatList[k] = new_Cmat

                    for k in range(0,len(mask_list_mol)):
                        new_mask = np.zeros((1, maxDim_nodes))
                        new_mask[0, :feat.shape[0]] = mask_list_mol[k]
                        mask_list_mol[k] = new_mask

                    new_Nmat_all = np.zeros((maxDim_links, maxDim_nodes))
                    new_Nmat_all[:Nall_mol.shape[0], :Nall_mol.shape[1]] = Nall_mol

                    new_Cmat_all = np.zeros((maxDim_links, maxDim_nodes))
                    new_Cmat_all[:Call_mol.shape[0], :Call_mol.shape[1]] = Call_mol

                    new_mask = np.zeros((1, maxDim_nodes))
                    new_mask[0, :feat.shape[0]] = mask

                else:  # NO PADDING
                    new_feat = feat
                    new_mask = mask
                    new_Nmat_all = Nall_mol
                    new_Cmat_all = Call_mol
                nm1, nm2, nm3, nm4 = NmatList
                cm1, cm2, cm3, cm4 = CmatList
                mk1,mk2,mk3,mk4 = mask_list_mol

                feat_list.append(new_feat)

                nm_list1.append(nm1)
                nm_list2.append(nm2)
                nm_list3.append(nm3)
                nm_list4.append(nm4)

                cm_list1.append(cm1)
                cm_list2.append(cm2)
                cm_list3.append(cm3)
                cm_list4.append(cm4)

                mask_list.append(new_mask)

                mask_list1.append(mk1)
                mask_list2.append(mk2)
                mask_list3.append(mk3)
                mask_list4.append(mk4)

                nmall_list.append(new_Nmat_all)
                cmall_list.append(new_Cmat_all)

            feat_batch = np.stack(feat_list)

            nm1_batch = np.stack(nm_list1)
            nm2_batch = np.stack(nm_list2)
            nm3_batch = np.stack(nm_list3)
            nm4_batch = np.stack(nm_list4)

            cm1_batch = np.stack(cm_list1)
            cm2_batch = np.stack(cm_list2)
            cm3_batch = np.stack(cm_list3)
            cm4_batch = np.stack(cm_list4)

            mask_batch = np.stack(mask_list)

            mask_batch1 = np.stack(mask_list1)
            mask_batch2 = np.stack(mask_list2)
            mask_batch3 = np.stack(mask_list3)
            mask_batch4 = np.stack(mask_list4)

            nmall_batch = np.stack(nmall_list)
            cmall_batch = np.stack(cmall_list)

            # registro le operazioni runnate durante il forward
            # in questo modo consento l'auto-differentiation
            with tf.GradientTape() as tape:
                # forward pass
                y_pred = tf.squeeze(model((feat_batch, nm1_batch, cm1_batch,
                                           nm2_batch, cm2_batch,
                                           nm3_batch, cm3_batch,
                                           nm4_batch, cm4_batch,
                                           mask_batch1, mask_batch2,
                                           mask_batch3, mask_batch4,
                                           nmall_batch, cmall_batch,
                                           mask_batch), training=True))
                # loss
                we = weight_generator.getWeigths(y_real[:, np.newaxis])
                loss_value = loss(y_real, y_pred, we)
            # gradiente parametri rispetto la loss
            grads = tape.gradient(loss_value, model.trainable_weights)
            grads, _ = tf.clip_by_global_norm(grads, 5.0)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            if y_pred.numpy().ndim == 0:
                y_pred = tf.reshape(y_pred, shape=(1,))
            train_metric.update_state(y_real, y_pred,sample_weight=we)
            if (idx + 1) % BATCH_VERBOSITY == 0:
                train_metric_printable = train_metric.result()
                print('Epoch', str(epoch + 1), '- after', str(idx + 1), 'batches training loss is:',
                      str(float(train_metric_printable)), file=logfile)
        tEnd = time.time()
        train_metric_printable = train_metric.result()
        buffer_tloss.append(train_metric.result().numpy())
        train_metric.reset_states()
        val_loss = perform_validationPPB(model, validation, val_metric, batchsize=BATCHSIZE, nfeat=NFEAT)
        buffer_vloss.append(val_loss)
        print('End of Epoch', str(epoch + 1), '- time:', str(round(tEnd - tStart, 2)), '- training loss:',
              str(float(train_metric_printable)), '- val loss: ', str(float(val_loss)), file=logfile)
        out_msg = eval_saving_wt(model, nameModel, buffer_vloss)
        if not (out_msg is None):
            print('Epoch ', str(epoch + 1), ': ', out_msg, file=logfile)
    return buffer_tloss, buffer_vloss



def perform_trainingCL(model,train,weight_generator,NEPOCHS,BATCHSIZE,optimizer,train_metric,validation,
                   val_metric,BATCH_VERBOSITY,NFEAT,nameModel,logfile):
    buffer_tloss = list()
    buffer_vloss = list()

    for epoch in range(NEPOCHS):
        tStart = time.time()
        print('----- EPOCH', str(epoch + 1), '/', str(NEPOCHS), '-----', file=logfile)
        # shuffling
        train = train.sample(frac=1, random_state=12345).reset_index(drop=True)
        # gen batches
        batch_train = gen_batches(len(train), BATCHSIZE)
        for idx, b in enumerate(batch_train):
            bt = train[b]
            feat_list = list()
            nm_list1 = list()
            nm_list2 = list()
            nm_list3 = list()
            nm_list4 = list()
            cm_list1 = list()
            cm_list2 = list()
            cm_list3 = list()
            cm_list4 = list()
            mask_list = list()
            y_real = bt.Y.to_numpy()
            if BATCHSIZE > 1:
                maxDim_nodes, maxDim_links = get_maxDims(bt, NFEAT)
            for smile in bt.SMILES:
                feat, NmatList, CmatList, mask = gen_smiles2graph(smile, NFEAT)

                if BATCHSIZE > 1:  # DO PADDING

                    new_feat = np.zeros((maxDim_nodes, feat.shape[1]))  # dimensione spazio latente
                    new_feat[:feat.shape[0], :feat.shape[1]] = feat

                    for k in range(0, len(NmatList)):
                        new_Nmat = np.zeros((maxDim_links, maxDim_nodes))
                        new_Nmat[:NmatList[k].shape[0], :NmatList[k].shape[1]] = NmatList[k]
                        NmatList[k] = new_Nmat

                    for k in range(0, len(CmatList)):
                        new_Cmat = np.zeros((maxDim_links, maxDim_nodes))
                        new_Cmat[:CmatList[k].shape[0], :CmatList[k].shape[1]] = CmatList[k]
                        CmatList[k] = new_Cmat

                    new_mask = np.zeros((1, maxDim_nodes))
                    new_mask[0, :feat.shape[0]] = mask

                else:  # NO PADDING
                    new_feat = feat
                    new_mask = mask
                nm1, nm2, nm3, nm4 = NmatList
                cm1, cm2, cm3, cm4 = CmatList

                feat_list.append(new_feat)

                nm_list1.append(nm1)
                nm_list2.append(nm2)
                nm_list3.append(nm3)
                nm_list4.append(nm4)

                cm_list1.append(cm1)
                cm_list2.append(cm2)
                cm_list3.append(cm3)
                cm_list4.append(cm4)

                mask_list.append(new_mask)

            feat_batch = np.stack(feat_list)

            nm1_batch = np.stack(nm_list1)
            nm2_batch = np.stack(nm_list2)
            nm3_batch = np.stack(nm_list3)
            nm4_batch = np.stack(nm_list4)

            cm1_batch = np.stack(cm_list1)
            cm2_batch = np.stack(cm_list2)
            cm3_batch = np.stack(cm_list3)
            cm4_batch = np.stack(cm_list4)

            mask_batch = np.stack(mask_list)
            # registro le operazioni runnate durante il forward
            # in questo modo consento l'auto-differentiation
            with tf.GradientTape() as tape:
                # forward pass
                y_pred = tf.squeeze(model((feat_batch, nm1_batch, cm1_batch,
                                           nm2_batch, cm2_batch,
                                           nm3_batch, cm3_batch,
                                           nm4_batch, cm4_batch,
                                           mask_batch), training=True))
                # loss
                we = weight_generator.getWeigths(y_real[:, np.newaxis])
                loss_value = loss(y_real, y_pred, we)
            # gradiente parametri rispetto la loss
            grads = tape.gradient(loss_value, model.trainable_weights)
            grads, _ = tf.clip_by_global_norm(grads, 5.0)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            if y_pred.numpy().ndim == 0:
                y_pred = tf.reshape(y_pred, shape=(1,))
            train_metric.update_state(y_real, y_pred,sample_weight=we)
            if (idx + 1) % BATCH_VERBOSITY == 0:
                train_metric_printable = train_metric.result()
                print('Epoch', str(epoch + 1), '- after', str(idx + 1), 'batches training loss is:',
                      str(float(train_metric_printable)), file=logfile)
        tEnd = time.time()
        train_metric_printable = train_metric.result()
        buffer_tloss.append(train_metric.result().numpy())
        train_metric.reset_states()
        val_loss = perform_validationCL(model, validation, val_metric, batchsize=BATCHSIZE, nfeat=NFEAT)
        buffer_vloss.append(val_loss)
        print('End of Epoch', str(epoch + 1), '- time:', str(round(tEnd - tStart, 2)), '- training loss:',
              str(float(train_metric_printable)), '- val loss: ', str(float(val_loss)), file=logfile)
        out_msg = eval_saving_wt(model, nameModel, buffer_vloss)
        if not (out_msg is None):
            print('Epoch ', str(epoch + 1), ': ', out_msg, file=logfile)
    return buffer_tloss, buffer_vloss

def perform_training_class(model,train,weights,trainLoss,NEPOCHS,BATCHSIZE,optimizer,train_metric,validation,
                   val_metric,BATCH_VERBOSITY,NFEAT,nameModel,logfile):
    buffer_tloss = list()
    buffer_vloss = list()
    for epoch in range(NEPOCHS):
        tStart = time.time()
        print('----- EPOCH', str(epoch + 1), '/', str(NEPOCHS), '-----', file=logfile)
        # shuffling
        train = train.sample(frac=1, random_state=12345).reset_index(drop=True)
        # gen batches
        batch_train = gen_batches(len(train), BATCHSIZE)
        for idx, b in enumerate(batch_train):
            bt = train[b]
            feat_list = list()
            nm_list1 = list()
            nm_list2 = list()
            nm_list3 = list()
            nm_list4 = list()
            cm_list1 = list()
            cm_list2 = list()
            cm_list3 = list()
            cm_list4 = list()
            mask_list = list()
            mask_list1 = list()
            mask_list2 = list()
            mask_list3 = list()
            mask_list4 = list()
            nmall_list = list()
            cmall_list = list()
            y_real = bt.Y.to_numpy()
            if BATCHSIZE > 1:
                maxDim_nodes, maxDim_links = get_maxDims(bt, NFEAT)
            for smile in bt.SMILES:
                feat, NmatList, CmatList, mask, mask_list_mol,Nall_mol,Call_mol = gen_smiles2graph(smile, NFEAT)

                if BATCHSIZE > 1:  # DO PADDING

                    new_feat = np.zeros((maxDim_nodes, feat.shape[1]))  # dimensione spazio latente
                    new_feat[:feat.shape[0], :feat.shape[1]] = feat



                    for k in range(0, len(NmatList)):
                        new_Nmat = np.zeros((maxDim_links, maxDim_nodes))
                        new_Nmat[:NmatList[k].shape[0], :NmatList[k].shape[1]] = NmatList[k]
                        NmatList[k] = new_Nmat

                    for k in range(0, len(CmatList)):
                        new_Cmat = np.zeros((maxDim_links, maxDim_nodes))
                        new_Cmat[:CmatList[k].shape[0], :CmatList[k].shape[1]] = CmatList[k]
                        CmatList[k] = new_Cmat

                    for k in range(0,len(mask_list_mol)):
                        new_mask = np.zeros((1, maxDim_nodes))
                        new_mask[0, :feat.shape[0]] = mask_list_mol[k]
                        mask_list_mol[k] = new_mask

                    new_Nmat_all = np.zeros((maxDim_links, maxDim_nodes))
                    new_Nmat_all[:Nall_mol.shape[0], :Nall_mol.shape[1]] = Nall_mol

                    new_Cmat_all = np.zeros((maxDim_links, maxDim_nodes))
                    new_Cmat_all[:Call_mol.shape[0], :Call_mol.shape[1]] = Call_mol

                    new_mask = np.zeros((1, maxDim_nodes))
                    new_mask[0, :feat.shape[0]] = mask

                else:  # NO PADDING
                    new_feat = feat
                    new_mask = mask
                    new_Nmat_all = Nall_mol
                    new_Cmat_all = Call_mol
                nm1, nm2, nm3, nm4 = NmatList
                cm1, cm2, cm3, cm4 = CmatList
                mk1,mk2,mk3,mk4 = mask_list_mol

                feat_list.append(new_feat)

                nm_list1.append(nm1)
                nm_list2.append(nm2)
                nm_list3.append(nm3)
                nm_list4.append(nm4)

                cm_list1.append(cm1)
                cm_list2.append(cm2)
                cm_list3.append(cm3)
                cm_list4.append(cm4)

                mask_list.append(new_mask)

                mask_list1.append(mk1)
                mask_list2.append(mk2)
                mask_list3.append(mk3)
                mask_list4.append(mk4)

                nmall_list.append(new_Nmat_all)
                cmall_list.append(new_Cmat_all)

            feat_batch = np.stack(feat_list)

            nm1_batch = np.stack(nm_list1)
            nm2_batch = np.stack(nm_list2)
            nm3_batch = np.stack(nm_list3)
            nm4_batch = np.stack(nm_list4)

            cm1_batch = np.stack(cm_list1)
            cm2_batch = np.stack(cm_list2)
            cm3_batch = np.stack(cm_list3)
            cm4_batch = np.stack(cm_list4)

            mask_batch = np.stack(mask_list)

            mask_batch1 = np.stack(mask_list1)
            mask_batch2 = np.stack(mask_list2)
            mask_batch3 = np.stack(mask_list3)
            mask_batch4 = np.stack(mask_list4)

            nmall_batch = np.stack(nmall_list)
            cmall_batch = np.stack(cmall_list)

            # registro le operazioni runnate durante il forward
            # in questo modo consento l'auto-differentiation
            with tf.GradientTape() as tape:
                # forward pass
                y_pred = tf.squeeze(model((feat_batch, nm1_batch, cm1_batch,
                                           nm2_batch, cm2_batch,
                                           nm3_batch, cm3_batch,
                                           nm4_batch, cm4_batch,
                                           mask_batch1, mask_batch2,
                                           mask_batch3, mask_batch4,
                                           nmall_batch, cmall_batch,
                                           mask_batch), training=True))
                # loss
                we = y_real*weights[1]
                we[we==0] = weights[0]
                if y_pred.numpy().ndim == 0:
                    y_pred = tf.reshape(y_pred, shape=(1,))
                loss_value = trainLoss(y_real, y_pred,sample_weight=tf.constant(we,shape=(1,y_real.shape[0])))
            # gradiente parametri rispetto la loss
            grads = tape.gradient(loss_value, model.trainable_weights)
            grads, _ = tf.clip_by_global_norm(grads, 5.0)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            train_metric.update_state(y_real, y_pred,sample_weight=tf.constant(we,shape=(1,y_real.shape[0])))
            if (idx + 1) % BATCH_VERBOSITY == 0:
                train_metric_printable = train_metric.result()
                print('Epoch', str(epoch + 1), '- after', str(idx + 1), 'batches training loss is:',
                      str(float(train_metric_printable)), file=logfile)
        tEnd = time.time()
        train_metric_printable = train_metric.result()
        buffer_tloss.append(train_metric.result().numpy())
        train_metric.reset_states()
        val_loss = perform_validation_class(model, validation, val_metric, batchsize=BATCHSIZE, nfeat=NFEAT)
        buffer_vloss.append(val_loss)
        print('End of Epoch', str(epoch + 1), '- time:', str(round(tEnd - tStart, 2)), '- training loss:',
              str(float(train_metric_printable)), '- val loss: ', str(float(val_loss)), file=logfile)
        out_msg = eval_saving_wt_max(model, nameModel, buffer_vloss)
        if not (out_msg is None):
            print('Epoch ', str(epoch + 1), ': ', out_msg, file=logfile)
    return buffer_tloss, buffer_vloss

def eval_saving_wt_max(model,nameModel,buffer_vloss):

    """
    Function that evaluates if model weights should be saved according to the latest value of the loss on the validation
    INPUT:
    - model: object representing the model
    - nameModel: string with the name of the model
    - buffer_vloss: list with the loss values computed in the epochs
    """


    performed_ep = len(buffer_vloss)
    if performed_ep == 0:
        return None
    elif performed_ep == 1:
        model.save_weights(nameModel+'.hdf5')
        return 'val loss improved from -inf to '+str(buffer_vloss[-1])+' -- saving weights to '+nameModel+'.hdf5'
    else:
        if buffer_vloss[-1]>np.array(buffer_vloss)[0:performed_ep-1].max():
            model.save_weights(nameModel + '.hdf5')
            return 'val loss improved from '+ str(np.array(buffer_vloss)[0:performed_ep-1].max()) \
                   + ' to '+ str(buffer_vloss[-1]) +\
                   ' -- saving weights to '+nameModel+'.hdf5'
        else:
            return 'val loss did not improve'


def perform_training_class_mt(model,train,weights,trainLoss,NEPOCHS,BATCHSIZE,optimizer,train_metric,validation,
                   val_metric,BATCH_VERBOSITY,NFEAT,nameModel,logfile):
    buffer_tloss = list()
    buffer_vloss = list()
    buffer_tloss_eval = list()
    buffer_vloss_eval = list()
    for epoch in range(NEPOCHS):
        tStart = time.time()
        print('----- EPOCH', str(epoch + 1), '/', str(NEPOCHS), '-----', file=logfile)
        # shuffling
        train = train.sample(frac=1, random_state=12345).reset_index(drop=True)
        # gen batches
        batch_train = gen_batches(len(train), BATCHSIZE)
        for idx, b in enumerate(batch_train):
            bt = train[b]
            feat_list = list()
            nm_list1 = list()
            nm_list2 = list()
            nm_list3 = list()
            nm_list4 = list()
            cm_list1 = list()
            cm_list2 = list()
            cm_list3 = list()
            cm_list4 = list()
            mask_list = list()
            mask_list1 = list()
            mask_list2 = list()
            mask_list3 = list()
            mask_list4 = list()
            nmall_list = list()
            cmall_list = list()
            y1_real = bt.Y1.to_numpy()
            y2_real = bt.Y2.to_numpy()
            y3_real = bt.Y3.to_numpy()
            if BATCHSIZE > 1:
                maxDim_nodes, maxDim_links = get_maxDims(bt, NFEAT)
            for smile in bt.SMILES:
                feat, NmatList, CmatList, mask, mask_list_mol,Nall_mol,Call_mol = gen_smiles2graph(smile, NFEAT)

                if BATCHSIZE > 1:  # DO PADDING

                    new_feat = np.zeros((maxDim_nodes, feat.shape[1]))  # dimensione spazio latente
                    new_feat[:feat.shape[0], :feat.shape[1]] = feat



                    for k in range(0, len(NmatList)):
                        new_Nmat = np.zeros((maxDim_links, maxDim_nodes))
                        new_Nmat[:NmatList[k].shape[0], :NmatList[k].shape[1]] = NmatList[k]
                        NmatList[k] = new_Nmat

                    for k in range(0, len(CmatList)):
                        new_Cmat = np.zeros((maxDim_links, maxDim_nodes))
                        new_Cmat[:CmatList[k].shape[0], :CmatList[k].shape[1]] = CmatList[k]
                        CmatList[k] = new_Cmat

                    for k in range(0,len(mask_list_mol)):
                        new_mask = np.zeros((1, maxDim_nodes))
                        new_mask[0, :feat.shape[0]] = mask_list_mol[k]
                        mask_list_mol[k] = new_mask

                    new_Nmat_all = np.zeros((maxDim_links, maxDim_nodes))
                    new_Nmat_all[:Nall_mol.shape[0], :Nall_mol.shape[1]] = Nall_mol

                    new_Cmat_all = np.zeros((maxDim_links, maxDim_nodes))
                    new_Cmat_all[:Call_mol.shape[0], :Call_mol.shape[1]] = Call_mol

                    new_mask = np.zeros((1, maxDim_nodes))
                    new_mask[0, :feat.shape[0]] = mask

                else:  # NO PADDING
                    new_feat = feat
                    new_mask = mask
                    new_Nmat_all = Nall_mol
                    new_Cmat_all = Call_mol
                nm1, nm2, nm3, nm4 = NmatList
                cm1, cm2, cm3, cm4 = CmatList
                mk1,mk2,mk3,mk4 = mask_list_mol

                feat_list.append(new_feat)

                nm_list1.append(nm1)
                nm_list2.append(nm2)
                nm_list3.append(nm3)
                nm_list4.append(nm4)

                cm_list1.append(cm1)
                cm_list2.append(cm2)
                cm_list3.append(cm3)
                cm_list4.append(cm4)

                mask_list.append(new_mask)

                mask_list1.append(mk1)
                mask_list2.append(mk2)
                mask_list3.append(mk3)
                mask_list4.append(mk4)

                nmall_list.append(new_Nmat_all)
                cmall_list.append(new_Cmat_all)

            feat_batch = np.stack(feat_list)

            nm1_batch = np.stack(nm_list1)
            nm2_batch = np.stack(nm_list2)
            nm3_batch = np.stack(nm_list3)
            nm4_batch = np.stack(nm_list4)

            cm1_batch = np.stack(cm_list1)
            cm2_batch = np.stack(cm_list2)
            cm3_batch = np.stack(cm_list3)
            cm4_batch = np.stack(cm_list4)

            mask_batch = np.stack(mask_list)

            mask_batch1 = np.stack(mask_list1)
            mask_batch2 = np.stack(mask_list2)
            mask_batch3 = np.stack(mask_list3)
            mask_batch4 = np.stack(mask_list4)

            nmall_batch = np.stack(nmall_list)
            cmall_batch = np.stack(cmall_list)

            # registro le operazioni runnate durante il forward
            # in questo modo consento l'auto-differentiation
            with tf.GradientTape(persistent=True) as tape:
                # forward pass
                y1_pred,y2_pred,y3_pred = tf.squeeze(model((feat_batch, nm1_batch, cm1_batch,
                                           nm2_batch, cm2_batch,
                                           nm3_batch, cm3_batch,
                                           nm4_batch, cm4_batch,
                                           mask_batch1, mask_batch2,
                                           mask_batch3, mask_batch4,
                                           nmall_batch, cmall_batch,
                                           mask_batch), training=True))
                # losses
                #Y1
                we1 = y1_real*weights[0][1]
                we1[we1==0] = weights[0][0]
                loss_value1 = trainLoss[0](y1_real, y1_pred,sample_weight=tf.constant(we1,shape=(1,y1_real.shape[0])))
                # Y2
                we2 = y2_real * weights[1][1]
                we2[we2 == 0] = weights[1][0]
                loss_value2 = trainLoss[1](y2_real, y2_pred, sample_weight=tf.constant(we2, shape=(1, y2_real.shape[0])))
                # Y3
                we3 = y3_real * weights[2][1]
                we3[we3 == 0] = weights[2][0]
                loss_value3 = trainLoss[2](y3_real, y3_pred, sample_weight=tf.constant(we3, shape=(1, y3_real.shape[0])))
            # gradiente parametri rispetto la loss
            grads1 = tape.gradient(loss_value1, model.trainable_weights)
            grads1, _ = tf.clip_by_global_norm(grads1, 5.0)
            optimizer.apply_gradients(zip(grads1, model.trainable_weights))

            grads2 = tape.gradient(loss_value2, model.trainable_weights)
            grads2, _ = tf.clip_by_global_norm(grads2, 5.0)
            optimizer.apply_gradients(zip(grads2, model.trainable_weights))

            grads3 = tape.gradient(loss_value3, model.trainable_weights)
            grads3, _ = tf.clip_by_global_norm(grads3, 5.0)
            optimizer.apply_gradients(zip(grads3, model.trainable_weights))
            del tape
            if y1_pred.numpy().ndim == 0:
                y1_pred = tf.reshape(y1_pred, shape=(1,))
                y2_pred = tf.reshape(y2_pred, shape=(1,))
                y3_pred = tf.reshape(y3_pred, shape=(1,))
            train_metric[0].update_state(y1_real, y1_pred,sample_weight=tf.constant(we1,shape=(1,y1_real.shape[0])))
            train_metric[1].update_state(y2_real, y2_pred, sample_weight=tf.constant(we2, shape=(1, y2_real.shape[0])))
            train_metric[2].update_state(y3_real, y3_pred, sample_weight=tf.constant(we3, shape=(1, y3_real.shape[0])))
            if (idx + 1) % BATCH_VERBOSITY == 0:
                train_metric_printable = train_metric.result()
                print('Epoch', str(epoch + 1), '- after', str(idx + 1), 'batches training loss is:',
                      str(float(train_metric_printable)), file=logfile)
        tEnd = time.time()
        train_metric_out = np.array([tm.result().numpy() for tm in train_metric])
        train_metric_printable = train_metric_out.max()
        buffer_tloss.append(train_metric_out)
        buffer_tloss_eval.append(train_metric_printable)
        for tm in train_metric:
            tm.reset_states()
        val_loss = perform_validation_class_mt(model, validation, val_metric, batchsize=BATCHSIZE, nfeat=NFEAT)
        buffer_vloss.append(val_loss)
        buffer_vloss_eval.append(np.array(val_loss).min())
        print('End of Epoch', str(epoch + 1), '- time:', str(round(tEnd - tStart, 2)), '- training loss:',
              str(float(train_metric_printable)), '- val loss: ', str(float(np.array(val_loss).min())), file=logfile)
        out_msg = eval_saving_wt_max(model, nameModel, buffer_vloss_eval)
        if not (out_msg is None):
            print('Epoch ', str(epoch + 1), ': ', out_msg, file=logfile)
    return buffer_tloss, buffer_vloss, buffer_tloss_eval, buffer_vloss_eval