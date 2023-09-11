#!/usr/bin/env python
#####################################################################################
# DESCRIPTION:
#####################################################################################
#These functions allow to compute metrics on activation values correlating with sparsity (Gini index and OCA) 

#####################################################################################
# LIBRAIRIES:
#####################################################################################
#public librairies
import numpy as np
import pandas
import matplotlib.pyplot as plt
import matplotlib as mpl
import sparsenesslib.plots as plots
import itertools
import time
import statsmodels.api as sm
import scipy.optimize as opt
import os
from sklearn.manifold import MDS
from scipy.ndimage import gaussian_filter1d
from scipy import linalg
from scipy.spatial import distance
from scipy import stats

from sklearn import decomposition
from sklearn.decomposition import IncrementalPCA
from sklearn import preprocessing
from sklearn import metrics
from joblib import dump, load
import csv


#####################################################################################
# PROCEDURES/FUNCTIONS:
#####################################################################################
def gini(vector):
    '''
    Compute Gini coefficient of an iterable object (lis, np.array etc)
    '''    
    if np.amin(vector) < 0:
        # Values cannot be negative:
        vector -= np.amin(vector)
    # Values cannot be 0:     
    vector = [i+0.0000001 for i in vector]     
    # Values must be sorted:
    vector = np.sort(vector)
    # Index per array element:
    index = np.arange(1,vector.shape[0]+1)
    # Number of array elements:
    n = vector.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * vector)) / (n * np.sum(vector)))

#####################################################################################
def acp_layers(dict_metrics, pc, bdd, layer, pathData = "../../", saveModele = False):
    
    '''
    A PCA with activations of each layer as features
    '''
    
    df_metrics = pandas.DataFrame.from_dict(dict_metrics)     
    tic = time.perf_counter()    

    for index, row in df_metrics.iterrows():
        
        n_comp = 10 
        
        df = pandas.DataFrame.from_dict(dict(zip(row.index, row.values))).T  
        X = df.values
        
        
        pc, pca = do_PCA(X)
        
       
        
        os.makedirs(pathData, exist_ok=True)
        pc.to_csv(pathData+"/"+"pca_values_"+layer+".csv")
        
        if saveModele:            
            os.makedirs(pathData+"/Modele", exist_ok=True)
            dump(pca, pathData+"/Modele"+"/"+layer+'_pca_model.joblib')



        
        toc = time.perf_counter()
        print(f"time: {toc - tic:0.4f} seconds")        

        getVarienceRatio(pca,bdd, layer, pathData)

#####################################################################################
def acp_layers_featureMap(dict_metrics, pc, bdd, layer, pathData = "../../", saveModele = False):
    
    '''
    A PCA with activations of each layer as features
    '''
    
    df_metrics = pandas.DataFrame.from_dict(dict_metrics)

    tic = time.perf_counter()

    for index, row in df_metrics.iterrows():
        coordFeature = np.empty([0,row.shape[0]])
        for n, _ in enumerate(row.values[0]):
            
            n_comp = 10 
            
            reshape = np.array(row.values.tolist())
            featureMap = reshape[:,n]
            df = pandas.DataFrame.from_dict(dict(zip(row.index,featureMap))).T  
            X = df.values
            pc, pca = do_PCA(X)
            if saveModele:
                
                os.makedirs(pathData+"/Modele", exist_ok=True)
                dump(pca, pathData+"/Modele"+"/"+layer+'_'+str(n) +'_pca_model.joblib')
                
            std_scale = preprocessing.StandardScaler().fit(pc)       
            coordinates_scaled = std_scale.transform(pc).T
            
            if n == 0:
                coordFeature = coordinates_scaled
            else:
                coordFeature = np.append(coordFeature, coordinates_scaled, 0)
        coordFeature = coordFeature.T
        pc , pca= do_PCA(coordFeature)

        print(bdd," show the bdd" )
        os.makedirs(pathData+"", exist_ok=True)        
        pc.to_csv(pathData+"/"+"pca_values_"+layer+".csv")

        if saveModele:            
            os.makedirs(pathData+"/Modele", exist_ok=True)
            dump(pca, pathData+"/Modele"+"/"+layer+'_pca_model.joblib')

        print('############################################################################')
        toc = time.perf_counter()
        print(f"time: {toc - tic:0.4f} seconds")
        print('############################################################################')

        getVarienceRatio(pca,bdd, layer, pathData)


############################################################################
def do_PCA(X, modele= None):
    
    std_scale = preprocessing.StandardScaler().fit(X)       
    X_scaled = std_scale.transform(X)
            
    if modele ==None:
        pca = decomposition.PCA(n_components= 0.8)
        pca.fit(X_scaled)
    else:
        pca = modele
        
    coordinates = pca.transform(X_scaled)
        
    return pandas.DataFrame(coordinates),pca








