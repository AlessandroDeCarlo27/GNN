import tensorflow as tf
import numpy as np
from utils.ProcessData import get_maxDims, gen_smiles2graph
from sklearn.utils import gen_batches
import matplotlib.pyplot as plt
import pandas as pd

def perform_validation(model,validation,val_metric,batchsize,nfeat):

    """
    Test the model on a validation set.
    INPUT:
    - model: model object
    - validation: Pandas Dataframe which represents validation set
    - val_metric: Metric object that represent the score function used for the evaluation
    - batchsize: dimension of batches
    - nfeat: number of features
    OUTPUT:
    - numpy value of the metric of interest on the validation set
    """



    val_metric.reset_states()
    batch_val = gen_batches(len(validation), batchsize)
    for b_val in batch_val:
        bv = validation[b_val]
        feat_list_val = list()
        nm_list1_val = list()
        nm_list2_val = list()
        nm_list3_val = list()
        nm_list4_val = list()

        cm_list1_val = list()
        cm_list2_val = list()
        cm_list3_val = list()
        cm_list4_val = list()

        mask_list_val = list()
        mask_list_val1 = list()
        mask_list_val2 = list()
        mask_list_val3 = list()
        mask_list_val4 = list()

        nmall_val = list()
        cmall_val = list()

        y_real_val = bv.Y.to_numpy()
        maxDim_nodes_val, maxDim_links_val = get_maxDims(bv,nfeat)
        for smile_val in bv.SMILES:
            feat_val, NmatList_val, CmatList_val, mask_val, mask_list_mol,Nall_mol,Call_mol = gen_smiles2graph(smile_val,nfeat)
            if batchsize>1: #PADDING

                new_feat_val = np.zeros((maxDim_nodes_val, feat_val.shape[1]))  # dimensione spazio latente
                new_feat_val[:feat_val.shape[0], :feat_val.shape[1]] = feat_val

                for k in range(0,len(NmatList_val)):
                    new_Nmat_val = np.zeros((maxDim_links_val, maxDim_nodes_val))
                    new_Nmat_val[:NmatList_val[k].shape[0],:NmatList_val[k].shape[1]] = NmatList_val[k]
                    NmatList_val[k] = new_Nmat_val

                for k in range(0,len(CmatList_val)):
                    new_Cmat_val = np.zeros((maxDim_links_val, maxDim_nodes_val))
                    new_Cmat_val[:CmatList_val[k].shape[0], :CmatList_val[k].shape[1]] = CmatList_val[k]
                    CmatList_val[k] = new_Cmat_val

                for k in range(0, len(mask_list_mol)):
                    new_mask = np.zeros((1, maxDim_nodes_val))
                    new_mask[0, :feat_val.shape[0]] = mask_list_mol[k]
                    mask_list_mol[k] = new_mask

                new_mask_val = np.zeros((1, maxDim_nodes_val))
                new_mask_val[0, :feat_val.shape[0]] = mask_val

                new_Nmat_all = np.zeros((maxDim_links_val, maxDim_nodes_val))
                new_Nmat_all[:Nall_mol.shape[0], :Nall_mol.shape[1]] = Nall_mol

                new_Cmat_all = np.zeros((maxDim_links_val, maxDim_nodes_val))
                new_Cmat_all[:Call_mol.shape[0], :Call_mol.shape[1]] = Call_mol
            else: #NO PADDING
                new_feat_val = feat_val
                new_mask_val = mask_val
                new_Nmat_all = Nall_mol
                new_Cmat_all = Call_mol

            nm1_val, nm2_val, nm3_val, nm4_val = NmatList_val
            cm1_val, cm2_val, cm3_val, cm4_val = CmatList_val
            mk1, mk2, mk3, mk4 = mask_list_mol

            feat_list_val.append(new_feat_val)
            mask_list_val.append(new_mask_val)


            nm_list1_val.append(nm1_val)
            nm_list2_val.append(nm2_val)
            nm_list3_val.append(nm3_val)
            nm_list4_val.append(nm4_val)

            cm_list1_val.append(cm1_val)
            cm_list2_val.append(cm2_val)
            cm_list3_val.append(cm3_val)
            cm_list4_val.append(cm4_val)

            mask_list_val1.append(mk1)
            mask_list_val2.append(mk2)
            mask_list_val3.append(mk3)
            mask_list_val4.append(mk4)

            nmall_val.append(new_Nmat_all)
            cmall_val.append(new_Cmat_all)

        feat_batch_val = np.stack(feat_list_val)
        nm_batch1_val = np.stack(nm_list1_val)
        nm_batch2_val = np.stack(nm_list2_val)
        nm_batch3_val = np.stack(nm_list3_val)
        nm_batch4_val = np.stack(nm_list4_val)

        cm_batch1_val = np.stack(cm_list1_val)
        cm_batch2_val = np.stack(cm_list2_val)
        cm_batch3_val = np.stack(cm_list3_val)
        cm_batch4_val = np.stack(cm_list4_val)

        mask_batch_val = np.stack(mask_list_val)

        mask_batch_val1 = np.stack(mask_list_val1)
        mask_batch_val2 = np.stack(mask_list_val2)
        mask_batch_val3 = np.stack(mask_list_val3)
        mask_batch_val4 = np.stack(mask_list_val4)

        nmall_batch = np.stack(nmall_val)
        cmall_batch = np.stack(cmall_val)

        y_pred_val = tf.squeeze(model((feat_batch_val,
                                       nm_batch1_val,cm_batch1_val,
                                       nm_batch2_val, cm_batch2_val,
                                       nm_batch3_val, cm_batch3_val,
                                       nm_batch4_val, cm_batch4_val,
                                       mask_batch_val1, mask_batch_val2,
                                       mask_batch_val3, mask_batch_val4,
                                       nmall_batch, cmall_batch,
                                       mask_batch_val), training=False))
        if y_pred_val.ndim == 0:
            y_pred_val = tf.reshape(y_pred_val,shape=(1,))
        val_metric.update_state(y_real_val,y_pred_val)
    return val_metric.result().numpy()


def load_best_model(model,namemodel):

    """
    Function that allow to load the best model according to val loss.
    INPUT:
    - model: model object with the same architecture of the model to be loaded
    - namemodel: name of the model whose weights need to be loaded
    OUTPUT:
    - model: input model with the best weights
    """

    name_w = namemodel+'.hdf5'
    model.load_weights(name_w)
    return model


def perform_test(model,validation,batchsize,nfeat):

    """
    Test the model on a test set. The batch schema was mantained in order to leverage GPU when huge test sets
    are used.
    INPUT:
    - model: model object
    - validation: Pandas Dataframe which represents test set
    - batchsize: dimension of batches
    - nfeat: number of features
    OUTPUT:
    - numpy of dim 1 with the predictions of the model on the test set
    """



    pred_list = list()
    batch_val = gen_batches(len(validation), batchsize)
    for b_val in batch_val:
        bv = validation[b_val]
        feat_list_val = list()
        nm_list1_val = list()
        nm_list2_val = list()
        nm_list3_val = list()
        nm_list4_val = list()

        cm_list1_val = list()
        cm_list2_val = list()
        cm_list3_val = list()
        cm_list4_val = list()

        mask_list_val = list()
        mask_list_val1 = list()
        mask_list_val2 = list()
        mask_list_val3 = list()
        mask_list_val4 = list()

        nmall_val = list()
        cmall_val = list()

        y_real_val = bv.Y.to_numpy()
        maxDim_nodes_val, maxDim_links_val = get_maxDims(bv, nfeat)
        for smile_val in bv.SMILES:
            feat_val, NmatList_val, CmatList_val, mask_val, mask_list_mol, Nall_mol, Call_mol = gen_smiles2graph(
                smile_val, nfeat)
            if batchsize > 1:  # PADDING

                new_feat_val = np.zeros((maxDim_nodes_val, feat_val.shape[1]))  # dimensione spazio latente
                new_feat_val[:feat_val.shape[0], :feat_val.shape[1]] = feat_val

                for k in range(0, len(NmatList_val)):
                    new_Nmat_val = np.zeros((maxDim_links_val, maxDim_nodes_val))
                    new_Nmat_val[:NmatList_val[k].shape[0], :NmatList_val[k].shape[1]] = NmatList_val[k]
                    NmatList_val[k] = new_Nmat_val

                for k in range(0, len(CmatList_val)):
                    new_Cmat_val = np.zeros((maxDim_links_val, maxDim_nodes_val))
                    new_Cmat_val[:CmatList_val[k].shape[0], :CmatList_val[k].shape[1]] = CmatList_val[k]
                    CmatList_val[k] = new_Cmat_val

                for k in range(0, len(mask_list_mol)):
                    new_mask = np.zeros((1, maxDim_nodes_val))
                    new_mask[0, :feat_val.shape[0]] = mask_list_mol[k]
                    mask_list_mol[k] = new_mask

                new_mask_val = np.zeros((1, maxDim_nodes_val))
                new_mask_val[0, :feat_val.shape[0]] = mask_val

                new_Nmat_all = np.zeros((maxDim_links_val, maxDim_nodes_val))
                new_Nmat_all[:Nall_mol.shape[0], :Nall_mol.shape[1]] = Nall_mol

                new_Cmat_all = np.zeros((maxDim_links_val, maxDim_nodes_val))
                new_Cmat_all[:Call_mol.shape[0], :Call_mol.shape[1]] = Call_mol
            else:  # NO PADDING
                new_feat_val = feat_val
                new_mask_val = mask_val
                new_Nmat_all = Nall_mol
                new_Cmat_all = Call_mol

            nm1_val, nm2_val, nm3_val, nm4_val = NmatList_val
            cm1_val, cm2_val, cm3_val, cm4_val = CmatList_val
            mk1, mk2, mk3, mk4 = mask_list_mol

            feat_list_val.append(new_feat_val)
            mask_list_val.append(new_mask_val)

            nm_list1_val.append(nm1_val)
            nm_list2_val.append(nm2_val)
            nm_list3_val.append(nm3_val)
            nm_list4_val.append(nm4_val)

            cm_list1_val.append(cm1_val)
            cm_list2_val.append(cm2_val)
            cm_list3_val.append(cm3_val)
            cm_list4_val.append(cm4_val)

            mask_list_val1.append(mk1)
            mask_list_val2.append(mk2)
            mask_list_val3.append(mk3)
            mask_list_val4.append(mk4)

            nmall_val.append(new_Nmat_all)
            cmall_val.append(new_Cmat_all)

        feat_batch_val = np.stack(feat_list_val)
        nm_batch1_val = np.stack(nm_list1_val)
        nm_batch2_val = np.stack(nm_list2_val)
        nm_batch3_val = np.stack(nm_list3_val)
        nm_batch4_val = np.stack(nm_list4_val)

        cm_batch1_val = np.stack(cm_list1_val)
        cm_batch2_val = np.stack(cm_list2_val)
        cm_batch3_val = np.stack(cm_list3_val)
        cm_batch4_val = np.stack(cm_list4_val)

        mask_batch_val = np.stack(mask_list_val)

        mask_batch_val1 = np.stack(mask_list_val1)
        mask_batch_val2 = np.stack(mask_list_val2)
        mask_batch_val3 = np.stack(mask_list_val3)
        mask_batch_val4 = np.stack(mask_list_val4)

        nmall_batch = np.stack(nmall_val)
        cmall_batch = np.stack(cmall_val)

        y_pred_val = tf.squeeze(model((feat_batch_val,
                                       nm_batch1_val, cm_batch1_val,
                                       nm_batch2_val, cm_batch2_val,
                                       nm_batch3_val, cm_batch3_val,
                                       nm_batch4_val, cm_batch4_val,
                                       mask_batch_val1, mask_batch_val2,
                                       mask_batch_val3, mask_batch_val4,
                                       nmall_batch, cmall_batch,
                                       mask_batch_val), training=False),axis=[-1,-2])
        pred_list.append(y_pred_val)
    return tf.concat(pred_list,axis=0).numpy()


def scatterplot(y_real_test,y_pred_test,nameVar,idxFig):

    plt.rcParams.update({'font.size': 36})
    plt.grid()
    plt.scatter(y_real_test, y_pred_test)
    idline = np.linspace(np.concatenate([y_real_test, y_pred_test]).min(),
                         np.concatenate([y_real_test, y_pred_test]).max(), 100)
    plt.plot(idline, idline, '-k', linewidth=5)
    fig = plt.gcf()
    fig.set_size_inches(32, 18)
    plt.title("Predictions vs Observations")
    plt.xlabel(nameVar + " obs.")
    plt.ylabel(nameVar + " pred")
    plt.savefig('PredVsObs'+str(idxFig)+'.png', bbox_inches='tight', dpi=250)
    plt.clf()

def residuals(y_real_test,res,nameVar,idxFig):
    plt.rcParams.update({'font.size': 36})
    plt.grid()
    plt.scatter(y_real_test, res)
    fig = plt.gcf()
    fig.set_size_inches(32, 18)
    plt.title("Observations vs Residual Errors")
    plt.xlabel(nameVar + " obs.")
    plt.ylabel("Residuals")
    plt.savefig('Res'+str(idxFig)+'.png', bbox_inches='tight', dpi=250)
    plt.clf()

def store_recap_test(buffer_rmse,buffer_mae,buffer_r2,filename):
    N = len(buffer_mae)
    idx_list = np.arange(1,N+1,1).tolist()
    if len(idx_list)==1:
        idx_list[0] = 'Hold Out'
    recap_df = pd.DataFrame(list(zip(idx_list,buffer_rmse,buffer_mae,buffer_r2)),
                            columns=['Fold','RMSE','MAE','R2'])

    recap_df.to_csv(filename, index=False)
    return recap_df


def perform_validationPPB(model,validation,val_metric,batchsize,nfeat):

    """
    Test the model on a validation set.
    INPUT:
    - model: model object
    - validation: Pandas Dataframe which represents validation set
    - val_metric: Metric object that represent the score function used for the evaluation
    - batchsize: dimension of batches
    - nfeat: number of features
    OUTPUT:
    - numpy value of the metric of interest on the validation set
    """



    val_metric.reset_states()
    batch_val = gen_batches(len(validation), batchsize)
    for b_val in batch_val:
        bv = validation[b_val]
        feat_list_val = list()
        nm_list1_val = list()
        nm_list2_val = list()
        nm_list3_val = list()
        nm_list4_val = list()

        cm_list1_val = list()
        cm_list2_val = list()
        cm_list3_val = list()
        cm_list4_val = list()

        mask_list_val = list()
        mask_list_val1 = list()
        mask_list_val2 = list()
        mask_list_val3 = list()
        mask_list_val4 = list()

        nmall_val = list()
        cmall_val = list()

        y_real_val = bv.Y.to_numpy()
        maxDim_nodes_val, maxDim_links_val = get_maxDims(bv,nfeat)
        for smile_val in bv.SMILES:
            feat_val, NmatList_val, CmatList_val, mask_val, mask_list_mol,Nall_mol,Call_mol = gen_smiles2graph(smile_val,nfeat)
            if batchsize>1: #PADDING

                new_feat_val = np.zeros((maxDim_nodes_val, feat_val.shape[1]))  # dimensione spazio latente
                new_feat_val[:feat_val.shape[0], :feat_val.shape[1]] = feat_val

                for k in range(0,len(NmatList_val)):
                    new_Nmat_val = np.zeros((maxDim_links_val, maxDim_nodes_val))
                    new_Nmat_val[:NmatList_val[k].shape[0],:NmatList_val[k].shape[1]] = NmatList_val[k]
                    NmatList_val[k] = new_Nmat_val

                for k in range(0,len(CmatList_val)):
                    new_Cmat_val = np.zeros((maxDim_links_val, maxDim_nodes_val))
                    new_Cmat_val[:CmatList_val[k].shape[0], :CmatList_val[k].shape[1]] = CmatList_val[k]
                    CmatList_val[k] = new_Cmat_val

                for k in range(0, len(mask_list_mol)):
                    new_mask = np.zeros((1, maxDim_nodes_val))
                    new_mask[0, :feat_val.shape[0]] = mask_list_mol[k]
                    mask_list_mol[k] = new_mask

                new_mask_val = np.zeros((1, maxDim_nodes_val))
                new_mask_val[0, :feat_val.shape[0]] = mask_val

                new_Nmat_all = np.zeros((maxDim_links_val, maxDim_nodes_val))
                new_Nmat_all[:Nall_mol.shape[0], :Nall_mol.shape[1]] = Nall_mol

                new_Cmat_all = np.zeros((maxDim_links_val, maxDim_nodes_val))
                new_Cmat_all[:Call_mol.shape[0], :Call_mol.shape[1]] = Call_mol
            else: #NO PADDING
                new_feat_val = feat_val
                new_mask_val = mask_val
                new_Nmat_all = Nall_mol
                new_Cmat_all = Call_mol

            nm1_val, nm2_val, nm3_val, nm4_val = NmatList_val
            cm1_val, cm2_val, cm3_val, cm4_val = CmatList_val
            mk1, mk2, mk3, mk4 = mask_list_mol

            feat_list_val.append(new_feat_val)
            mask_list_val.append(new_mask_val)


            nm_list1_val.append(nm1_val)
            nm_list2_val.append(nm2_val)
            nm_list3_val.append(nm3_val)
            nm_list4_val.append(nm4_val)

            cm_list1_val.append(cm1_val)
            cm_list2_val.append(cm2_val)
            cm_list3_val.append(cm3_val)
            cm_list4_val.append(cm4_val)

            mask_list_val1.append(mk1)
            mask_list_val2.append(mk2)
            mask_list_val3.append(mk3)
            mask_list_val4.append(mk4)

            nmall_val.append(new_Nmat_all)
            cmall_val.append(new_Cmat_all)

        feat_batch_val = np.stack(feat_list_val)
        nm_batch1_val = np.stack(nm_list1_val)
        nm_batch2_val = np.stack(nm_list2_val)
        nm_batch3_val = np.stack(nm_list3_val)
        nm_batch4_val = np.stack(nm_list4_val)

        cm_batch1_val = np.stack(cm_list1_val)
        cm_batch2_val = np.stack(cm_list2_val)
        cm_batch3_val = np.stack(cm_list3_val)
        cm_batch4_val = np.stack(cm_list4_val)

        mask_batch_val = np.stack(mask_list_val)

        mask_batch_val1 = np.stack(mask_list_val1)
        mask_batch_val2 = np.stack(mask_list_val2)
        mask_batch_val3 = np.stack(mask_list_val3)
        mask_batch_val4 = np.stack(mask_list_val4)

        nmall_batch = np.stack(nmall_val)
        cmall_batch = np.stack(cmall_val)

        y_pred_val = tf.squeeze(model((feat_batch_val,
                                       nm_batch1_val,cm_batch1_val,
                                       nm_batch2_val, cm_batch2_val,
                                       nm_batch3_val, cm_batch3_val,
                                       nm_batch4_val, cm_batch4_val,
                                       mask_batch_val1, mask_batch_val2,
                                       mask_batch_val3, mask_batch_val4,
                                       nmall_batch, cmall_batch,
                                       mask_batch_val), training=False))
        if y_pred_val.ndim == 0:
            y_pred_val = tf.reshape(y_pred_val,shape=(1,))
        val_metric.update_state(np.power(y_real_val, 0.25), np.power(y_pred_val, 0.25))
    return val_metric.result().numpy()

def perform_validationCL(model,validation,val_metric,batchsize,nfeat):

    """
    Test the model on a validation set.
    INPUT:
    - model: model object
    - validation: Pandas Dataframe which represents validation set
    - val_metric: Metric object that represent the score function used for the evaluation
    - batchsize: dimension of batches
    - nfeat: number of features
    OUTPUT:
    - numpy value of the metric of interest on the validation set
    """



    val_metric.reset_states()
    batch_val = gen_batches(len(validation), batchsize)
    for b_val in batch_val:
        bv = validation[b_val]
        feat_list_val = list()
        nm_list1_val = list()
        nm_list2_val = list()
        nm_list3_val = list()
        nm_list4_val = list()

        cm_list1_val = list()
        cm_list2_val = list()
        cm_list3_val = list()
        cm_list4_val = list()

        mask_list_val = list()
        y_real_val = bv.Y.to_numpy()
        maxDim_nodes_val, maxDim_links_val = get_maxDims(bv,nfeat)
        for smile_val in bv.SMILES:
            feat_val, NmatList_val, CmatList_val, mask_val = gen_smiles2graph(smile_val,nfeat)
            if batchsize>1: #PADDING

                new_feat_val = np.zeros((maxDim_nodes_val, feat_val.shape[1]))  # dimensione spazio latente
                new_feat_val[:feat_val.shape[0], :feat_val.shape[1]] = feat_val

                for k in range(0,len(NmatList_val)):
                    new_Nmat_val = np.zeros((maxDim_links_val, maxDim_nodes_val))
                    new_Nmat_val[:NmatList_val[k].shape[0],:NmatList_val[k].shape[1]] = NmatList_val[k]
                    NmatList_val[k] = new_Nmat_val

                for k in range(0,len(CmatList_val)):
                    new_Cmat_val = np.zeros((maxDim_links_val, maxDim_nodes_val))
                    new_Cmat_val[:CmatList_val[k].shape[0], :CmatList_val[k].shape[1]] = CmatList_val[k]
                    CmatList_val[k] = new_Cmat_val

                new_mask_val = np.zeros((1, maxDim_nodes_val))
                new_mask_val[0, :feat_val.shape[0]] = mask_val
            else: #NO PADDING
                new_feat_val = feat_val
                new_mask_val = mask_val

            nm1_val, nm2_val, nm3_val, nm4_val = NmatList_val
            cm1_val, cm2_val, cm3_val, cm4_val = CmatList_val

            feat_list_val.append(new_feat_val)
            mask_list_val.append(new_mask_val)


            nm_list1_val.append(nm1_val)
            nm_list2_val.append(nm2_val)
            nm_list3_val.append(nm3_val)
            nm_list4_val.append(nm4_val)

            cm_list1_val.append(cm1_val)
            cm_list2_val.append(cm2_val)
            cm_list3_val.append(cm3_val)
            cm_list4_val.append(cm4_val)

        feat_batch_val = np.stack(feat_list_val)
        nm_batch1_val = np.stack(nm_list1_val)
        nm_batch2_val = np.stack(nm_list2_val)
        nm_batch3_val = np.stack(nm_list3_val)
        nm_batch4_val = np.stack(nm_list4_val)

        cm_batch1_val = np.stack(cm_list1_val)
        cm_batch2_val = np.stack(cm_list2_val)
        cm_batch3_val = np.stack(cm_list3_val)
        cm_batch4_val = np.stack(cm_list4_val)

        mask_batch_val = np.stack(mask_list_val)
        y_pred_val = tf.squeeze(model((feat_batch_val,
                                       nm_batch1_val,cm_batch1_val,
                                       nm_batch2_val, cm_batch2_val,
                                       nm_batch3_val, cm_batch3_val,
                                       nm_batch4_val, cm_batch4_val,
                                       mask_batch_val), training=False))
        if y_pred_val.ndim == 0:
            y_pred_val = tf.reshape(y_pred_val,shape=(1,))
        val_metric.update_state(np.power(10,y_real_val),np.power(10,y_pred_val))
    return val_metric.result().numpy()
