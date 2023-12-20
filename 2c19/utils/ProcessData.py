import rdkit, rdkit.Chem, rdkit.Chem.rdDepictor, rdkit.Chem.Draw
import networkx as nx
from rdkit import Chem
import pandas as pd
import numpy as np

def filterValidSmiles(dataset):
    """
    Function used to remove invalid SMILES from dataset. An non-valid SMILE is a SMILE which cannot be correctly parsed
    by rdkit tools i.e., output object of the parsing is a None
    INPUT:
    - dataset: must be a pandas dataframe with a column named SMILES which contains the strings of the SMILES
    OUTPUT:
    - dataset: input dataset without the non-valid SMILES
    - index_list: list with the indices of the valid SMILES
    - invalid_list: list with the indices of the non-valid SMILES
    """

    mols_obj = dataset["SMILES"].apply(Chem.MolFromSmiles)
    index_list = list()
    invalid_list = list()
    # Remove invalid smiles
    for index, value in mols_obj.items():
        if value is not None:
            index_list.append(index)
        else:
            invalid_list.append(index)

    # select only molecules with a valid smile
    dataset = dataset.iloc[index_list].reset_index(drop=True)

    return dataset, index_list, invalid_list

def one_hot_encoding(x, permitted_list):

    """
    Function that allows to convert a categorical value into a one-hot encoding representation
    INPUT:
    - x: current value
    - permitted_list: list of the possible values
    OUTPUT:
    - binary_encoding: one hot representation of the categorical value
    """

    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
    return binary_encoding


def get_atom_features(atom):
    """
    Function that extracts the features for each atom. We have adopted a feature vector in which:
    - each categorical feature is represented with one-hot encoding
    - for binary features we adopted a 1-0 coding
    INPUT:
    - atom: rdkit atom object
    OUTPUT:
    - np.array with atom features
    """
    element = one_hot_encoding(int(atom.GetAtomicNum()), list(range(1, 101)))
    # compute atom features
    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()),
                                              ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])

    is_in_a_ring_enc = [int(atom.IsInRing())]
    is_aromatic_enc = [int(atom.GetIsAromatic())]
    chirality_enc = one_hot_encoding(str(atom.GetChiralTag()),
                                      ['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW', 'CHI_OTHER'])

    nhs = [0]*5
    nhs[atom.GetTotalNumHs()] = 1

    atom_feature_vector = element + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + chirality_enc + nhs
    return np.array(atom_feature_vector)


def gen_smiles2graph(sml: object, features: object) -> object:
    """
    Function that extracts the features of a graph starting from the SMILE of a molecule.
    INPUT:
    - sml: string object with the smile
    OUTPUT:
    - features: list with
                - nodes: matrix #nodes x #features
                - Nmat_list: list of matrix, len= bond type considered. For each bond type, we provide a one-hot
                             representation of the neighbours in the links
                - Cmat_list: list of matrix, len= bond type considered. For each bond type, we provide a one-hot
                             representation of the center in the links
                - mask: one hot np.array for detecting atoms of a graph
    """
    m = rdkit.Chem.MolFromSmiles(sml)
    order_string = {
        rdkit.Chem.rdchem.BondType.SINGLE: 1,
        rdkit.Chem.rdchem.BondType.DOUBLE: 2,
        rdkit.Chem.rdchem.BondType.TRIPLE: 3,
        rdkit.Chem.rdchem.BondType.AROMATIC: 4,
    }
    N = len(list(m.GetAtoms()))

    nodes = np.zeros((N, features))
    for i in m.GetAtoms():
        nodes[i.GetIdx()] = get_atom_features(i)

    adj = np.zeros((4, N, N))
    for j in m.GetBonds():
        u = min(j.GetBeginAtomIdx(), j.GetEndAtomIdx())
        v = max(j.GetBeginAtomIdx(), j.GetEndAtomIdx())
        order = j.GetBondType()
        if order in order_string:
            order = order_string[order]
        else:
            raise Warning("Ignoring bond order" + order)
        adj[order-1,u, v] = 1
        adj[order - 1,v, u] = 1

    adj_all = np.sum(adj,axis=0)
    adj_all += np.eye(N)
    adj_all[adj_all>1]==1

    #N and C overall
    g_all = nx.from_numpy_matrix(adj_all, create_using=nx.DiGraph)
    src_tgt_all = nx.to_pandas_edgelist(g_all)[['source', 'target']].to_numpy()
    Nmat_all = np.zeros((src_tgt_all.shape[0], N))  # neighbor matrix
    Nmat_all[np.arange(0, src_tgt_all.shape[0]), src_tgt_all[:, 1]] = 1

    Cmat_all = np.zeros((src_tgt_all.shape[0], N))  # center matrix
    Cmat_all[np.arange(0, src_tgt_all.shape[0]), src_tgt_all[:, 0]] = 1


    adj += np.eye(N)
    mask = np.ones((1,N)) #mask for the whole graph
    Nmat_list = list()
    Cmat_list = list()
    mask_list = list()
    for k in range(0,4):
        adj_k = adj[k,:,:]
        g = nx.from_numpy_matrix(adj_k, create_using=nx.DiGraph)
        src_tgt = nx.to_pandas_edgelist(g)[['source', 'target']].to_numpy()
        Nmat = np.zeros((src_tgt.shape[0], N)) #neighbor matrix
        Nmat[np.arange(0, src_tgt.shape[0]), src_tgt[:, 1]] = 1
        Nmat[:,Nmat.sum(axis=-2)==1] = 0
        mask_i = (Nmat.sum(axis=-2)>0).astype(float) #mask for each bond type
        mask_list.append(mask_i)

        Cmat = np.zeros((src_tgt.shape[0], N)) #center matrix
        Cmat[np.arange(0, src_tgt.shape[0]), src_tgt[:, 0]] = 1
        Cmat[:, Cmat.sum(axis=-2) == 1] = 0
        Nmat_list.append(Nmat)
        Cmat_list.append(Cmat)
    return nodes, Nmat_list, Cmat_list, mask,mask_list,Nmat_all,Cmat_all # Here adj has the self loop inside


def get_maxDims(soldata,NFEAT):
    """
    Given a set of graph, compute the maximum number of links and nodes within the set
    INPUT:
    - soldata: Pandas Dataframe with the set of graphs
    - NFEAT: number of features of each graph
    OUTPUT:
    - maxDim_nodes: massimo numero di nodi di un grafo
    - maxDim_links: massimo numer di links di un grafo
    """

    maxDim_nodes = 0
    maxDim_links = 0
    for smile in soldata.SMILES:
        nodes, _, _, _,_,Nmatall,_ = gen_smiles2graph(smile,NFEAT)
        maxDim_nodes=max(maxDim_nodes, nodes.shape[0])
        maxDim_links = max(maxDim_links, Nmatall.shape[0])

    return maxDim_nodes, maxDim_links


def k_fold_scaffolds(dataset,Nfolds,seed):
    dataset = dataset.sample(frac=1,random_state=seed).reset_index(drop=True)
    smiles = dataset.SMILES.to_list()
    y = dataset.Y.to_numpy()
    xs = np.zeros(len(smiles))
    dcdat = dc.data.DiskDataset.from_numpy(X=xs, y=y, ids=smiles)
    scaffoldsplitter = dc.splits.ScaffoldSplitter()
    list_folds = scaffoldsplitter.k_fold_split(dcdat, Nfolds)
    return list_folds

def from_dc_to_pd_dataset(dc_data):
    pd_data = dc_data.to_dataframe().drop(labels=["X", "w"], axis=1)
    pd_data = pd_data.rename(columns={"ids": "SMILES", "y": "Y"})
    return pd_data


def get_PD_test(fold_asset):
    _, tst = fold_asset
    return from_dc_to_pd_dataset(tst)

def get_PD_train_val(fold_asset,seed):
    tv, _ = fold_asset
    scaffoldsplitter = dc.splits.ScaffoldSplitter()
    train, val = scaffoldsplitter.train_test_split(dataset=tv, frac_train=0.8, seed=seed)
    return from_dc_to_pd_dataset(train), from_dc_to_pd_dataset(val)


