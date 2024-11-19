import os
import scanpy as sc
import pandas as pd
import anndata as ad
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import issparse

import pickle

# label_dict = {0: 'normal', 1: 'tumour'}

os.chdir("/scratch/sah2p/datasets/GSE131907")

TrainData = ad.read_h5ad('./Alldata_hvg.h5ad')

idTmps = []
dataList = list()
dataLabel = list()
cell_ids = [] 

for idTmp in TrainData.obs['sample'].cat.categories:
    print(idTmp)
    idTmps.append(idTmp)
    aDataTmp = TrainData[TrainData.obs['sample'] == idTmp]
    # Ensure that aDataTmp.X is converted to a dense NumPy array
    if issparse(aDataTmp.X):  # Check if the data is a sparse matrix
        dataCell = torch.FloatTensor(aDataTmp.X.toarray())  # Convert sparse to dense and then to torch tensor
    else:
        dataCell = torch.FloatTensor(np.array(aDataTmp.X))  # Ensure it's a NumPy array and convert to torch tensor

    print(dataCell.shape)    
    dataList.append(dataCell)

    cell_ids.extend(aDataTmp.obs.index) 
    
    if aDataTmp.obs['phenotype'].values[0] == 'normal':
        dataLabel.append(0)
    elif aDataTmp.obs['phenotype'].values[0] == 'tumour':
        dataLabel.append(1)
    
        
df = pd.DataFrame(idTmps, columns=["sample"])
df.to_csv("./sampleid_560.csv", index=False)    

df_cell_ids = pd.DataFrame(cell_ids, columns=["cell_id"])
df_cell_ids.to_csv('./cell_id_560.csv', index=False)
   
assert len(dataLabel) == len(dataList)
print(len(dataLabel))

# Build Datasets
lb =  LabelEncoder()
dataLabels = lb.fit_transform(dataLabel)
dataLabels = torch.Tensor(dataLabels)

with open('./GSE131907_DataList_560_1213.pickle', 'wb') as handle:
    pickle.dump(dataList, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./GSE131907_DataLabel_560_1213.pickle', 'wb') as handle:
    pickle.dump(dataLabels, handle, protocol=pickle.HIGHEST_PROTOCOL)

