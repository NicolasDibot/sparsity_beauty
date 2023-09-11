#!/usr/bin/env python
#####################################################################################
# DESCRIPTION:
#####################################################################################
#The metrics can be computed in several ways on the layers, these different functions prtmettent to apply these different ways
#A mathematical formalization, with formulas, will be produced to describe them formally

#1. compute_filter: Compute chosen formula (gini, treve-rolls, l1 norm...) for each filter (flatten), and compute the mean of them

#2. compute_flatten: Create a flatten vector from each layer and compute chosen formula (gini, treve-rolls, l1 norm...) on it

#3. compute_channel: Compute chosen formula (gini, treve-rolls, l1 norm...) for each channel (1D vector on z dimension), and 
# compute the mean of them

#4. compute_activations: Executes one of the 3 previous functions according to the approach passed in parameter

#####################################################################################
# LIBRAIRIES:
#####################################################################################
#public librairies
from numpy.linalg import norm
import statistics as st
import scipy
import sys
import numpy as np
#personnal librairies
sys.path.insert(1,'../../code/functions')
import sparsenesslib.metrics as metrics
import sparsenesslib.entropy as etp

#####################################################################################
# PROCEDURES/FUNCTIONS:
#####################################################################################
def compute_filter(activations, activations_dict,layer,formula,k):
    '''
    Compute chosen formula (gini, treve-rolls, l1 norm...) for each filter (flatten), and compute the mean of them
    '''    
    filter = []

    if layer[0:5] == 'input':
        layer = 'input' + '_' + str(k)

    tuple = activations[layer].shape
    liste = list(tuple)    
    nb_channels = liste[3] 


    #maps is a new shape of activations, to have each mpas for first iteration, useful to compute entropy and complexity 
    maps = np.copy(activations[layer][0])  
    maps = np.moveaxis(maps,-1,0)

    entropies = []
    complexity = []

    #print('##### current layer is: ', layer)

    for map in maps:
        _, _, probabilities = etp.get_ordinal_probabilities(map,2,2)
        S = etp.Shannon_entropy(probabilities)
        
    for each in range(0, nb_channels-1): 
        filter.append([])
        index_row = 0
        for row in activations[layer][0]:
            index_column = 0
            for column in activations[layer][0][index_row]:                            
                filter[each].append(activations[layer][0][index_row][index_column][each])            
                index_column += 1
            index_row += 1

    filter_metrics = []

    for each in filter:
        if formula == 'L0':                
            filter_metrics.append(1 - (norm(each, 0)/len(each)))
        elif formula == 'L1':                
            filter_metrics.append(norm(each, 1))
        elif formula == 'treve-rolls':
            filter_metrics.append(metrics.treves_rolls(each))
        elif formula == 'gini':            
            filter_metrics.append(metrics.gini(each))
        elif formula == 'hoyer':            
            filter_metrics.append(metrics.hoyer(each))
        elif formula == 'kurtosis':
            filter_metrics.append(scipy.stats.kurtosis(each))
        else: print('ERROR: formula setting isnt L0, L1, treve-rolls, hoyer, gini or kurtosis')
    activations_dict[layer] = st.mean(filter_metrics)

####################################################/length#################################
def compute_flatten(activations, activations_dict,layer,formula,k):
    '''
    Create a flatten vector from each layer and compute chosen formula (gini, treve-rolls, l1 norm...) on it"
    '''
    if layer[0:5] == 'input':
        layer = 'input' + '_' + str(k)
    
    arr = np.array([])

    for key, value in activations.items():   # iter on both keys and values
        if key.startswith(layer): #permet de travailler par layer, par block etc..
            arr2 = activations[key].flatten()
            arr = np.append(arr, arr2, 0)

    #arr = activations[layer].flatten()
    if formula == 'L0':
        activations_dict[layer] = (1 - (norm(arr, 0)/len(arr)))
    elif formula == 'L1':
        activations_dict[layer] = (norm(arr, 1))        
    elif formula == 'treve-rolls':        
        activations_dict[layer] = (metrics.treves_rolls(arr))
    elif formula == 'gini':
        activations_dict[layer] = (metrics.gini(arr))
    elif formula == 'hoyer':
        activations_dict[layer] = (metrics.hoyer(arr))
    elif formula == 'kurtosis':
        activations_dict[layer] = (scipy.stats.kurtosis(arr))
    elif formula == 'mean':
        #activations_dict[layer] = st.mean(arr) 
        activations_dict[layer] = sum(arr) / len(arr)
    elif formula == "max":
        activations_dict[layer] =  np.amax(arr)
    elif formula == 'acp':        
        activations_dict[layer] = arr
    else: print('ERROR: formula setting isnt L0, L1, treve-rolls, hoyer, gini, kurtosis, mean or acp')


def compute_flatten_byCarte(activations, activations_dict,layer,formula,k):
    '''
    Create a flatten vector from each layer and compute chosen formula (gini, treve-rolls, l1 norm...) on it"
    '''
    if layer[0:5] == 'input':
        layer = 'input' + '_' + str(k)
    
    

    for key, value in activations.items():   # iter on both keys and values
        if key.startswith(layer):
            
            shape = activations[key].shape
            if formula in ('mean', 'max'):
                allarr = np.empty(shape[3])
            else:
                allarr = np.empty([shape[3],shape[1]*shape[2]])

            for n in range(activations[key].shape[3]):
                arr = activations[key][:,:,:,n].flatten()
                
                if formula == 'L0':
                    activ = (1 - (norm(arr, 0)/len(arr)))
                elif formula == 'L1':    
                    activ = (norm(arr, 1))        
                elif formula == 'treve-rolls':        
                    activ = (metrics.treves_rolls(arr))
                elif formula == 'gini':
                    activ = (metrics.gini(arr))
                elif formula == 'hoyer':
                    activ = (metrics.hoyer(arr))
                elif formula == 'kurtosis':
                    activ = (scipy.stats.kurtosis(arr))
                elif formula == 'mean':
                    #activ = st.mean(arr) 
                    activ = sum(arr) / len(arr)
                elif formula == "max":
                    activ = np.amax(arr)
                elif formula == 'acp':
                    activ = arr

                else: print('ERROR: formula setting isnt L0, L1, treve-rolls, hoyer, gini, kurtosis, mean or acp')

                allarr[n] = activ
            activations_dict[layer] = allarr

#####################################################################################
def compute_channel(activations, activations_dict,layer,formula,k):
    '''
    Compute chosen formula (gini, treve-rolls, l1 norm...) for each channel, and compute the mean of them
    '''
    channels = []
    index_row = 0

    if layer[0:5] == 'input':
        layer = 'input' + '_' + str(k)

    for row in activations[layer][0]:
        index_column = 0
        for column in activations[layer][0][index_row]:            
            channel = activations[layer][0][index_row][index_column]      
            if formula == 'L0':                
                channels.append(1-(norm(channel, 0)/len(channel)))                
            elif formula == 'L1':                
                channels.append(norm(channel, 1))
            elif formula == 'treve-rolls':
                channels.append(metrics.treves_rolls(channel))
            elif formula == 'gini':
                channels.append(metrics.gini(channel))
            elif formula == 'hoyer':
                channels.append(metrics.hoyer(channel))
            elif formula == 'kurtosis':
                channels.append(scipy.stats.kurtosis(channel))
            else: print('ERROR: formula setting isnt L0, L1, treve-rolls, hoyer, gini or kurtosis')
            index_column += 1
        index_row += 1    
    activations_dict[layer] = st.mean(channels)
#####################################################################################
def compute_activations(layers, flatten_layers, computation, activations, activations_dict,formula, k):
    '''
    executes one of the 3 previous functions according to the approach passed in parameter
    '''        
    for layer in layers:            
        if computation == 'channel':                
            if layer in flatten_layers:
                compute_flatten(activations, activations_dict, layer,formula,k)       
            else:                     
                compute_channel(activations, activations_dict, layer, formula,k)
        elif computation == 'filter':
            if layer in flatten_layers:
                compute_flatten(activations, activations_dict, layer,formula,k)       
            else:                     
                compute_filter(activations, activations_dict, layer, formula,k)
        elif computation == 'flatten':
            compute_flatten(activations, activations_dict, layer, formula,k)                
        else: print('ERROR: computation setting isnt channel, filter or flatten')
#####################################################################################


