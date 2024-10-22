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

# 3. Preparation of the ANN

# Load the training, validation, and test datasets from the corresponding files
def load_data(device=DEVICE, bs=BS, pos=0):
    
    # Training data
    dfOneOutput = pd.DataFrame()
    valorOut=[]
    dataTrainingR = dataTraining.copy()
    for espectra in dataTrainingR["BinarSpectrum"]:
        cont = 0
        peak = 0
        finded = False    
        while ((cont < len(espectra))and(finded == False)):
            if(espectra[cont]== pos):
                peak = 1
                finded = True
            else:
                cont += 1
        valorOut = valorOut + [float(peak)]   
    dfOneOutput["Embedding"] = dataTrainingR["FinalVec"]
    dfOneOutput["Out"] = valorOut
    
    x1 = dfOneOutput["Embedding"].to_numpy() #convert dataframe column as a numpy type
    x2 = np.vstack(x1).astype(np.float32) #convert objects to numpy type from object to float
    
    y1 = dfOneOutput["Out"].to_numpy() #convert dataframe column as a numpy type
    y2 = np.vstack(y1).astype(np.float32) #convert objects to numpy type from object to float
    
    # Train set
    ds = TensorDataset(torch.from_numpy(x2), torch.from_numpy(y2))
    train_dataloader = DataLoader(ds, batch_size=bs)
    
    # Validation data
    dfOneOutputValidation = pd.DataFrame()
    valorOut=[]
    dataValidationR = dataValidation.copy()
    for espectra in dataValidationR["BinarSpectrum"]:
        cont = 0
        peak = 0
        finded = False    
        while ((cont < len(espectra))and(finded == False)):
            if(espectra[cont]== pos):
                peak = 1
                finded = True
            else:
                cont += 1
        valorOut = valorOut + [float(peak)]  
    dfOneOutputValidation["Embedding"] = dataValidationR["FinalVec"]
    dfOneOutputValidation["Out"] = valorOut
    
    vx1 = dfOneOutputValidation["Embedding"].to_numpy() #convert dataframe column as a numpy type
    vx2 = np.vstack(vx1).astype(np.float32) #convert objects to numpy type from object to float
    
    vy1 = dfOneOutputValidation["Out"].to_numpy() #convert dataframe column as a numpy type
    vy2 = np.vstack(vy1).astype(np.float32) #convert objects to numpy type from object to float
    
    # Validation set
    ds = TensorDataset(torch.from_numpy(vx2), torch.from_numpy(vy2))
    validation_dataloader = DataLoader(ds, batch_size=bs)
    
    #Test Data
    dfOneOutputTest = pd.DataFrame()
    valorOut=[]
    dataTestR = dataTest.copy()
    for espectra in dataTestR["BinarSpectrum"]:
        cont = 0
        peak = 0
        finded = False    
        while ((cont < len(espectra))and(finded == False)):
            if(espectra[cont]== pos):
                peak = 1
                finded = True
            else:
                cont += 1
        valorOut = valorOut + [float(peak)]    
    dfOneOutputTest["Embedding"] = dataTestR["FinalVec"]
    dfOneOutputTest["Out"] = valorOut
    
    tx1 = dfOneOutputTest["Embedding"].to_numpy() #convert dataframe column as a numpy type
    tx2 = np.vstack(tx1).astype(np.float32) #convert objects to numpy type from object to float
    
    ty1 = dfOneOutputTest["Out"].to_numpy() #convert dataframe column as a numpy type
    ty2 = np.vstack(ty1).astype(np.float32) #convert objects to numpy type from object to float
   
    # Test set
    ds = TensorDataset(torch.from_numpy(tx2), torch.from_numpy(ty2))
    test_dataloader = DataLoader(ds, batch_size=bs)
    # Done
    return train_dataloader, validation_dataloader, test_dataloader

# 4. ANN architecture

# The feedforward network class
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


# 5. Considerated Position

posP=dfMostFreq100.posPeak[conPeak]

# 6. Training

# Define train loop
def train_loop(dataloader, model, loss_fn, optimizer, device=DEVICE):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Done
    return

# Create and train a network
def create_train_ann(train_dataloader, validaton_dataloader, test_dataloader,
                     posP, max_epochs=100, min_epochs=0, lr=LR,
                     autostop=True, ns=100, quiet=False, nrep=1, device=DEVICE):
    val_loss_min, tr_losses, val_losses, te_loss = 1e10, [], [], -1
 
    # Repeat nrep times
    for rep in range(nrep):
        # Instantiate the network
        model = MLP().to(device)
        # Initialize the loss function (binary cross-entropy)
        loss_fn = nn.BCELoss()
        # Initialize the optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        # Train
        for t in range(max_epochs):
            # Fit
            train_loop(train_dataloader, model, loss_fn, optimizer, device=device)

            # Get training error
            x_tr = train_dataloader.dataset.tensors[0].to(device)
            y_tr = train_dataloader.dataset.tensors[1].to(device)
            tr_loss = loss_fn(model(x_tr), y_tr)
            
            # Get validation error
            with torch.no_grad():
                x_val = validation_dataloader.dataset.tensors[0].to(device)
                y_val = validation_dataloader.dataset.tensors[1].to(device)
                val_loss = loss_fn(model(x_val), y_val)
                if val_loss < val_loss_min:
                    val_loss_min = val_loss
                    # Save the best model from the validation set
                    modelVal=open("data/ANN/ModelVal{}.pickle".format(posP), "wb") 
                    pickle.dump(model, modelVal)
                    modelVal.close()
                    # Get test error
                    x_te = test_dataloader.dataset.tensors[0].to(device)
                    y_te = test_dataloader.dataset.tensors[1].to(device)
                    te_loss = float(loss_fn(model(x_te), y_te))

            # Reporting
            tr_losses.append(float(tr_loss))
            val_losses.append(float(val_loss))
            # Reporting
#             if not quiet and (t % 10) == 0:
#                 print(t, tr_losses[-1], val_losses[-1]) #, teloss)
#                     # Plot losses
#                 plt.plot(tr_losses)
#                 plt.plot(val_losses)
#                 plt.yscale('log')
#                 plt.show()
#                 plt.figure(figsize=(10, 6))

            if (t == (max_epochs-1)):
                plt.plot(tr_losses)
                plt.plot(val_losses)
                plt.yscale('log')
                plt.gca().legend(('Training losses', 'Validation losses'))
                plt.savefig("data/ANN/ANNgraphs/TrainingValidationCurve{}.jpg".format(posP), bbox_inches='tight')
                #save loss
                dfLosses = pd.DataFrame()
                dfLosses["training"] = tr_losses
                dfLosses["validation"] = val_losses
                saveLoss=open("data/ANN/ANNgraphs/Losses{}.pickle".format(posP), "wb") 
                pickle.dump(dfLosses, saveLoss)
                saveLoss.close()
            # Stop criterion
            if (autostop == True and
                t >= min_epochs and
                np.mean(val_losses[-ns:]) > np.mean(val_losses[-2*ns:-ns]) and
                np.mean(tr_losses[-ns:]) < np.mean(tr_losses[-2*ns:-ns])):
                plt.plot(tr_losses)
                plt.plot(val_losses)
                plt.yscale('log')
                plt.gca().legend(('Training losses', 'Validation losses'))
                plt.savefig("data/ANN/ANNgraphs/TrainingValidationCurve{}.jpg".format(posP), bbox_inches='tight')
                #save loss
                dfLosses = pd.DataFrame()
                dfLosses["training"] = tr_losses
                dfLosses["validation"] = val_losses
                saveLoss=open("data/ANN/ANNgraphs/Losses{}.pickle".format(posP), "wb") 
                pickle.dump(dfLosses, saveLoss)
                saveLoss.close()
                break

    # Done
    return model, tr_losses, val_losses, te_loss


# 7. Utilization of the ANN

dfResults = pd.DataFrame()  

# Create/load the data
train_dataloader, validation_dataloader, test_dataloader = load_data(pos=posP)

# Create the network and train it
model, tr_losses, val_losses, te_loss = create_train_ann(
        train_dataloader, validation_dataloader, test_dataloader, posP,
        max_epochs=2000, min_epochs=400, lr=LR, autostop=True, ns=100, nrep=1,
) #2000, 400

#Data set with the results
pred_specs = model(test_dataloader.dataset.tensors[0])
y_te = test_dataloader.dataset.tensors[1]
pred=pred_specs.detach().numpy()
sol=y_te.detach().numpy()
dfResults["Prediction"] = pred.tolist()
dfResults["Solution"] = sol
    
# Loop to save values
for cont, i in enumerate(dfResults["Prediction"]):
    p= str(i)[1:-1]#take off the brakets 
    dfResults["Prediction"][cont]=float(p)

#display(dfResults)

# Save data in a pickle
print("saved model", posP)
pickle_out=open("data/ANN/PredictionSolution{}.pickle".format(posP), "wb")
pickle.dump(dfResults, pickle_out)
pickle_out.close()

# Sleeping time to not colapse the clusters' nodes (when using cluster)
time.sleep(60) 
