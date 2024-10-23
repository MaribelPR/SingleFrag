# SingleFrag: A deep learning tool for MS/MS fragment and spectral prediction and metabolite annotation.

## Highlights

- Novel deep learning tool that predicts individual fragments separately rather than attempting to predict the entire fragmentation spectrum at once.
- Surpasses state-of-the-art fragmentation tools.

## Overview

Metabolite and small molecule identification via MS/MS involves matching experimental spectra with prerecorded spectra of known compounds. This process is hindered by the current lack of comprehensive reference spectral libraries. To address this gap, we need accurate *in silico* fragmentation tools for predicting MS/MS spectra of compounds without empirical data. Here we present an overview of SingleFrag and how our deep learning tool can be used to predict fragmentation spectra of molecules, considering individual fragments separately. 

### Authors

Maribel Pérez-Ribera<sup>a</sup>, Muhammad Faizan Khan^b^, Roger Giné^b^, Josep M. Badia^b^, Sandra
Junza^b^, Oscar Yanes^b,c^, Marta Sales-Pardo^a,*^, and Roger Guimerà^a,d,*^

^a^Department of Chemical Engineering, Universitat Rovira i Virgili, 43007 Tarragona, Catalonia
^b^Department of Electronic Engineering, IISPV, Universitat Rovira i Virgili, 43007 Tarragona, Catalonia
^c^CIBER de Diabetes y Enfermedades Metabólicas Asociadas (CIBERDEM), Instituto de Salud Carlos III, 28029 Madrid, Spain
^d^ICREA, 08010 Barcelona, Catalonia
^*^Corresponding authors: Marta Sales-Pardo (E-mail: marta.sales@urv.cat), Roger Guimerà (E-mail: roger.guimera@urv.cat)

## Usage

Here we present some instructions about how to use our tool and predict spectra from input molecules. 

### Installation

First, we have to set up the environment with all the necessarily packages. All of them are specified in the file *requirements.txt*. Also, we must use the version Python 3.8.19. 

### Data

The following step would be to download from **Fighare** our dataset called "SingleFrag", which contains the data files necessary to run the code. 

The way we recommend to store these files is:

    -> data/
            --> inputMols_git.xlsx (file containing the input molecules, already inside data.zip folder)

            --> ANN/
                    --> ModelVal*.pickle (trained ANN models corresponding to the most frequented positions in the training set spectra)
            --> GNN/
                    --> ModelVal*.pickle (trained GNN models corresponding to the most frequented positions in the training set spectra)
            --> COM/
                    --> ModelVal*.pickle (trained COM models corresponding to the most frequented positions in the training set spectra)

            --> Mol2vecModel/
                             --> Mol2vecModelGeneral_rep_dades1.model

            --> modelData/
                          --> AllMostFreqMolGeneral_rep_dades1.pickle
                          --> thresholdsANN.pickle
                          --> thresholdsGNN.pickle
                          --> thresholdsCOM.pickle
                          
    -> results/ (create this folder to save the predictions)

Regarding the input file format (*inputMols_git.xlsx*):

Inside the *data.zip* folder, we already find a toy example of an input file. This contains 4 columns:

smiles  \         name  \              spectrum  \            link

0  CCC(=O)C(O)=O  2-Ketobutyric acid   [27.0, 28.0, 29.0...   https://hmdb.ca/spectra/ms_ms/1473314 

Where the SMILES is the molecule format that we use as an input to our model, the NAME is how the molecule is called and its SPECTRUM is found in https://hmdb.ca/spectra/ms_ms/1473314 using MS/MS in positive mode. However, this is a toy example as our model is able to predict the fragments with more resolution (considering 2 decimals) and this spectrum format only contains integers. Additionally, what we only need as an input to our models is the SMILES of the molecule, so the rest information is complementary, except the SPECTRUM which we may need in case we want to compare the prediction of our tool with an experimental spectrum. 

Concerning the files from **Fighare**:

1. Neural networks trained to predict the presence of a peak in a specific MZ localization in a tandem mass spectrum. In the name of every file, the MZ position is written in the following way (mz*100).
    - ANN: Artificial Neural Networks
    - GNN: Graph Neural Networks
    - COM: Networks that combine the predictive power of ANN and GNN
2. Mol2vecModel: Contains a Mol2vec model trained to obtain a 300-dimensional vector from molecule SMILES (Mol2vecModelGeneral_rep_dades1.model).
3. modelData:
    - AllMostFreqMolGeneral_rep_dades1.pickle: file containing the number of peaks that are contained in every MZ bin from tandem mass spectra in the training set.
    - thresholdsANN.pickle: threshold per each of the most frequented 1,000 MZ positions in the training set. If a prediction using an ANN model for a specific position is higher or equal to this value (for its specific MZ position), means that a peak in that bin is predicted.
    - thresholdsGNN.pickle: same as above but for the GNN models.
    - thresholdsCOM.pickle: same as above but for the COM models.

Every *.pickle* file was created using Python 3.8.19.

### Spectra prediction

Once all the above steps have been done, we can proceed to predict the MS/MS spectra from molecules. To do this, we use the *spectraPrediction.ipynb*. 

In this file, we can find all the steps that we have to follow, in order to obtain a predicted MSMS spectrum for each of our input molecules. Furthermore, in case of having the experimental spectrum of a molecule, we can also find a graphical comparison with the predicted one.

Things to consider:
- In this file, we only consider the ANN models as they got better results than the GNN and the COM.
- The input to our ANN models is a 300 dimensions vector generated by applying the Mol2vec model to the input SMILES.
- Thinking about efficacy in case we wanted to predict the spectra of multiple molecules at the same time, we first create a file with all the predictions per MZ position (*results/dfPredictionPeak_***.pickle*), and then we reconstruct each spectra considering all the predictions from the different MZ models (*r'results/predSpectra.json'*). 

### Training the models

If we wanted to train the models, we would have to prepare a training, validation and test set, and use the available files: *ANNmodel.py*, *GNNmodel.py* and *COMmodel.py*. Each of this files corresponds to a different type of neural network, and they are prepared to predict if molecules are presenting peak in a specific MZ position that we will have to specify as input. Thus, to obtain the whole reconstruction of a molecule spectrum, we will have to predict the peaks in a desired set of MZ positions, and later on but them together. 


## 💭 Feedback and Contributing

TODO: Link to the Discussions tab in our repository (when Public)
