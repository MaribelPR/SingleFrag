#!/usr/bin/env python
# coding: utf-8

# LIBRARIES

#conda install anaconda=custom 

import os
import numpy as np
import pickle
import pandas as pd
import math
import argparse 
import warnings
warnings.filterwarnings("ignore")
import time

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='ticks', font_scale=1.5)


import torch
from torch import nn
from torch.utils.data import TensorDataset, Dataset, DataLoader
import torchvision

from statannot import add_stat_annotation
from scipy.stats import spearmanr

#DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'
print(f'Using {DEVICE} device')
##conda install torchvision -c pytorch

#Model metrics
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

# 1. Reading of parameters from comand window

parser = argparse.ArgumentParser()

parser.add_argument("-C","--conPeak",help="Numero de peaks",type=int)

args = parser.parse_args()

conPeak = args.conPeak

BS = 16     # Batch size for training
LR = 1.e-4  # Learning rate for training


# 2. Data preparation

# Training Dataset
pickle_in=open("TrainingDataSet.pickle", "rb")
dataTraining=pickle.load(pickle_in)

# Validation Dataset
pickle_in=open("ValidationDataSet.pickle", "rb")
dataValidation=pickle.load(pickle_in)

# Test Dataset
pickle_in=open("TestDataSet.pickle", "rb")
dataTest=pickle.load(pickle_in)

# MZ most frequented positions in training set
pickle_in=open("data/modelData/AllMostFreqMolGeneral_rep_dades1.pickle", "rb")
dfMostFreq100=pickle.load(pickle_in)
dfMostFreq100.reset_index(inplace=True, drop=True)#reset index

# 3. Preparation of the input embeddings
ENCEL = ['B', 'N', 'C', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'Ca', 'K', 'Na', 'Mg']
ENCBT = [
    Chem.rdchem.BondType.AROMATIC,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
]
NNFEAT = 15

# Edge embeddings
def get_edges(molecule, encbt=ENCBT):
    edges, attributes = [], []
    for bond in molecule.GetBonds():
        # Edge index
        edges.append([bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()])
        edges.append([bond.GetEndAtom().GetIdx(), bond.GetBeginAtom().GetIdx()])
        # Edge attributes (bond type, one-hot)
        this_attrib = [0] * len(encbt)
        for nbt, bt in enumerate(encbt):
            if bond.GetBondType() == bt:
                this_attrib[nbt] = 1
        attributes.append(this_attrib)
        attributes.append(this_attrib) # add twice for both directions
    # Done
    return torch.tensor(edges).transpose(0, 1), torch.Tensor(attributes)
    
# Node embeddings
def embed_atom(atom, encel=ENCEL):
    """
    1) See 'Comparison of Atom Representations in Graph Neural Networks for Molecular Property Prediction'.
    2) Add atom mass.
    """
    # Initialize (just in case this is necessary)
    embedding = []
    """# Element one-hot
    embedding = [0] * len(encel)
    try:
        embedding[encel.index(atom.GetSymbol())] = 1
    except ValueError:
        pass"""
    # Number of heavy neighbors one-hot
    subemb = [0] * 6 
    subemb[len(atom.GetNeighbors())] = 1
    embedding += subemb
    # Number of implicit Hs one-hot
    subemb = [0] * 5
    subemb[atom.GetNumImplicitHs()] = 1
    embedding += subemb
    # Formal charge
    embedding += [atom.GetFormalCharge()]
    # Is in ring
    embedding += [atom.IsInRing()]
    # Is aromatic
    embedding += [atom.GetIsAromatic()]
    # Atomic mass
    embedding += [atom.GetMass()]
    # Done
    return embedding

# Embed all atoms in a molecule
def embed_atoms(molecule, encel=ENCEL):
    return torch.Tensor([embed_atom(atom, encel=encel) for atom in molecule.GetAtoms()])

# 4. Position that we want to consider
posP=dfMostFreq100.posPeak[conPeak]

# 5. Preparation of the ANN
# Load the training, validation, and test datasets from the corresponding files
def load_data(Device=device, bs=BS, pos=0):

    # Prepare the whole dataset with all the embeddings
    # Training data
    datasetTrain = []
    dataTrainingR = dataTraining.copy()    
    for cont,espectra in enumerate(dataTrainingR["BinarSpectrum"]):
        #GNN:
        #y
        cont2 = 0
        peak = 0
        finded = False    
        while ((cont2 < len(espectra))and(finded == False)):
            if(espectra[cont2]== pos):
                peak = 1
                finded = True
            else:
                cont2 += 1 
        #x
        x = embed_atoms(dataTrainingR["mol"][cont])
        #edge_index and edge_attr
        edge_index, edge_attr = get_edges(dataTrainingR["mol"][cont])
        #ANN:
        xA = dataTrainingR["FinalVec"][cont]
        xA = np.vstack(xA).astype(np.float32)
        datasetTrain.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=float(peak), xA = torch.from_numpy(xA)).to(device))#.to(Device)) #to(device)-->to send it to cuda

    # Create the DataLoader
    train_dl = DataLoader(datasetTrain, batch_size = BS)
    
    # Validation data
    datasetVal = []
    dataValidationR = dataValidation.copy() 
    for cont,espectra in enumerate(dataValidationR["BinarSpectrum"]):
        #y
        cont2 = 0
        peak = 0
        finded = False    
        while ((cont2 < len(espectra))and(finded == False)):
            if(espectra[cont2]== pos):
                peak = 1
                finded = True
            else:
                cont2 += 1
        #x
        x = embed_atoms(dataValidationR["mol"][cont])
        #edge_index and edge_attr
        edge_index, edge_attr = get_edges(dataValidationR["mol"][cont])
        #ANN:
        #xA:
        xA = dataValidationR["FinalVec"][cont]
        xA = np.vstack(xA).astype(np.float32) 
        datasetVal.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=float(peak), xA = torch.from_numpy(xA)).to(device))#.to(Device))
                            
    # Create the DataLoader
    validation_dl = DataLoader(datasetVal, batch_size = BS)
    
    # Test data
    datasetTest = []
    dataTestR = dataTest.copy() 
    for cont,espectra in enumerate(dataTestR["BinarSpectrum"]):
        #y
        cont2 = 0
        peak = 0
        finded = False    
        while ((cont2 < len(espectra))and(finded == False)):
            if(espectra[cont2]== pos):
                peak = 1
                finded = True
            else:
                cont2 += 1
        #x
        x = embed_atoms(dataTestR["mol"][cont])
        #edge_index and edge_attr
        edge_index, edge_attr = get_edges(dataTestR["mol"][cont])
        #ANN:
        #xA:
        xA = dataTestR["FinalVec"][cont]
        xA = np.vstack(xA).astype(np.float32)
        datasetTest.append(Data(x = x, edge_index = edge_index, edge_attr = edge_attr, y = float(peak), xA = torch.from_numpy(xA)).to(device))#.to(Device))
                                  
    # Create the DataLoader
    test_dl = DataLoader(datasetTest, batch_size = BS)
    
    # Done
    return train_dl, validation_dl, test_dl

COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00' '#ffff33', '#a65628', '#f781bf']
NHEAD = 1

# 6. Network architectures
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(300, 300), #16,300 x 300,300
            nn.ReLU(),
            nn.Linear(300, 300), 
            nn.ReLU(),
            nn.Linear(300, 300), 
            nn.Sigmoid(),  #16,300 (same shape as the input)
        )
        self.linAnn =  nn.Linear(300, 45)

    def forward(self, x):
        xann = self.linear_relu_stack(x) #16,300  
        xann = self.linAnn(xann)  #(16, 45) 
        return xann

class GATN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(NNFEAT, NNFEAT, heads=NHEAD) 
        self.ff1 = Linear(NNFEAT*NHEAD, NNFEAT) 
        self.conv2 = GATConv(NNFEAT, NNFEAT, heads=NHEAD)
        self.ff2 = Linear(NNFEAT*NHEAD, NNFEAT)
        self.conv3 = GATConv(NNFEAT, NNFEAT, heads=NHEAD)
        self.ff3 = Linear(NNFEAT*NHEAD, NNFEAT)
        self.mlp1 = Linear(6*NNFEAT, 6*NNFEAT) #Linear(90, 90)
        self.mplClas = MLP()
        self.mlp2 = Linear(6*NNFEAT, 1)
        
    def forward(self, data):
        x, edge_index, edge_attr, xA = data.x, data.edge_index, data.edge_attr, data.xA
        batch = data.batch #BS=16
        x = self.conv1(x=x, edge_index=edge_index) #(460,15) x (15,15) (460,15)
        #print(x.shape)
        x = self.ff1(x)  #(460,15)
        x = F.relu(x) #(460,15)
        x = self.conv2(x=x, edge_index=edge_index) #(460,15)
        x = self.ff2(x) #(460,15)
        x = F.relu(x) #(460,15) 
        x = self.conv3(x=x, edge_index=edge_index)#(460,15)
        x = self.ff3(x) #(460,15)
        #reduction gnn
        x1 = global_add_pool(x, batch) #(16, 15)
        x2 = global_max_pool(x, batch) #(16, 15))
        #print(x2.shape)
        x3 = global_mean_pool(x, batch) #(16, 15)
        #print(x3.shape)
        x = torch.cat([x1, x2, x3], dim=1) #(16, 45)
        shBatch = x.shape[0]
        #print("batch: ", shBatch)
        xA = xA.reshape([shBatch, 300])
        xann = self.mplClas(xA)  #(16, 45)
        x = torch.cat([x, xann], dim=1) #(16, 90)
        x = self.mlp1(x) #(16, 90)
        x = F.relu(x) #(16, 90)
        x = self.mlp2(x) #(16, 90)x (90, 1)= (16, 1)
        return torch.sigmoid(x)  #(16, 1)
        
# 7. Create the model
model = GATN().to(device)
# Initialize the loss function (binary cross-entropy)
loss_fn = nn.BCELoss()

# 8. Training

def create_train_com(train_dl, validation_dl, test_dl, max_epochs=5000, min_epochs=400, lr=LR, autostop=True, ns=100, quiet=False, nrep=1, Device=device): 
    val_loss_min, trlosses, vlosses, teloss = 1e10, [], [], -1
    
     # Initialize the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr) 
   
    # Train
    for t in range(max_epochs): #number of epochs we want to go through
        # Fit (train_loop)
        trloss, trnorm = 0, 0
        for datab in train_dl: 
            datab.to(device) #we put the data in the gpu
            # Compute prediction and loss
            pred = model(datab) 
            loss = loss_fn(pred, torch.reshape(datab.y, pred.shape))
            trloss += float(loss)
            trnorm += 1
            
            # Backpropagation
            optimizer.zero_grad() #sets the gradien to 0 for all the variables. It
                #is very important because if you do not do this, by default, pythorch is going to accumulate your gradient.
            loss.backward() #here we compute the gradient for all the parameters
            optimizer.step() #for all these parameters that we already have a gradient, we want to perform an 
                #optimization step, with an specified learning rate.
        trlosses.append(trloss / trnorm)

        # Get validation error
        with torch.no_grad():
            vloss, vnorm = 0, 0
            vppeaks, vpprobs = [], []
            for vdatab in validation_dl:
                vdatab.to(device)
                vpred = model(vdatab)
                vpprobs += list(vpred.detach().cpu().numpy().flatten())
                vppeaks += list(vdatab.y.detach().cpu().numpy().flatten())
                vloss += loss_fn(vpred, torch.reshape(vdatab.y, vpred.shape)).item()
                vnorm += 1
            vlosses.append(vloss / vnorm)

	    
            if vloss < val_loss_min:
                val_loss_min = vloss
                # Save the best model from the validation set
                modelVal=open("data/COM/ModelVal{}.pickle".format(posP), "wb")
                pickle.dump(model, modelVal)
                modelVal.close()
    
        if (t == (max_epochs-1)):
            # Plot losses
            plt.plot(trlosses)
            plt.plot(vlosses)
            plt.yscale('log')
            plt.gca().legend(('Training losses', 'Validation losses'))
            plt.savefig("data/COM/COMgraphs/TrainingValidationCurve{}.jpg".format(posP), bbox_inches='tight')
            #save loss
            dfLosses = pd.DataFrame()
            dfLosses["training"] = trlosses
            dfLosses["validation"] = vlosses
            saveLoss=open("data/COM/COMgraphs/Losses{}.pickle".format(posP), "wb")
            pickle.dump(dfLosses, saveLoss)
            saveLoss.close()
            
            #AUTOSTOP
        if (autostop == True and
             t >= min_epochs and
             np.mean(vlosses[-ns:]) > np.mean(vlosses[-2*ns:-ns]) and
             np.mean(trlosses[-ns:]) < np.mean(trlosses[-2*ns:-ns])):
            plt.plot(trlosses)
            plt.plot(vlosses)
            plt.yscale('log')
            plt.gca().legend(('Training losses', 'Validation losses'))
            plt.savefig("data/COM/COMgraphs/TrainingValidationCurve{}.jpg".format(posP), bbox_inches='tight')
            #save loss
            dfLosses = pd.DataFrame()
            dfLosses["training"] = trlosses
            dfLosses["validation"] = vlosses
            saveLoss=open("data/COM/COMgraphs/Losses{}.pickle".format(posP), "wb")
            pickle.dump(dfLosses, saveLoss)
            saveLoss.close()
            break
            
     # Done
    return model, trlosses, vlosses#, teloss

numP = dfMostFreq100.nPeak[conPeak] 


# 9. Utilization of the network

dfResults = pd.DataFrame()  
# Create/load the data
train_dataloader, validation_dataloader, test_dataloader = load_data(
        Device=device, bs=BS, pos=posP
)

# Create the network and train it
model, tr_losses, val_losses = create_train_com(
        train_dataloader, validation_dataloader, test_dataloader,
        max_epochs=10000, min_epochs=4000, lr=LR, autostop=True, ns=100, quiet=False, nrep=1, Device=device
)

#Data set with the results
listPred=[]
listSol=[]
solModel, predModel=[],[]

for tdatab in test_dataloader:
    pred_specs = model(tdatab)
    y_te = tdatab.y
    pred = pred_specs.detach().cpu().numpy() 
    sol = y_te.detach().cpu().numpy()
    pred2 = pred.tolist()
    
    for pos in pred2:
        p = str(pos)[1:-1] #take off the brakets 
        listPred.append(float(p))
    sol2 = sol.tolist()
    
    for pos2 in sol2:
        listSol.append(float(pos2))
        
dfResults["Prediction"] = listPred
dfResults["Solution"] = listSol

# Save data in a pickle
pickle_out=open("data/COM/PredictionSolution{}.pickle".format(posP), "wb")
pickle.dump(dfResults, pickle_out)
pickle_out.close()

# Sleeping time to not colapse the clusters' nodes (when using cluster)
time.sleep(60) 

