#!/usr/bin/env python
#####################################################################################
# DESCRIPTION:
#####################################################################################
#[EN]These functions allow to compute metrics on activation values correlating with sparsity (Gini index, L0 norm etc) 
#and the "metametrics" allowing to characterize them (regressions on the curve of the metric according to the layer, distributions...)

#[FR]Ces fonctions permettent de calculer les métriques sur les valeurs d'activation corrélant avec la sparsité (indice de Gini, norme L0 etc) 
#ainsi que les méta métriques permettant de les caractériser (régressions sur la courbe de la métriue en fonction de la couche, distributions ...)

#1. gini: Compute Gini coefficient of an iterable object (lis, np.array etc)

#2. treve-rolls: Compute modified treve-rolls population sparsennes, formula from (wilmore et all, 2000)

#3. reglog: Compute a logistic regression for each picture between layer"s metric value (y) and number of layer (x)

#4. minmax: Compute for each picture the difference between the higher and lower value of layer's metrics 

#5. gompertzFct: Define the Gompertz function model

#6. reg_gompertz: Compute a regression on the Gompertz function. Function for the moment not functional. 

#7. histplot_metrics: Plot a histogram of the distribution of metrics for all images regardless of the layer in which they were calculated

#8. compress_metrics: If metrics values aren't between 0 and 1, like kurtosis or L0 norm, change theme like this to let them availaible 
#for logistic regression. 1 will be th highest value of the metric, 0 the lowest.

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


###############
def verbosePourcent(current, valMax):
    if verbosePourcent.pourcent < int(current*100/valMax):
        print("\033[A                             \033[A")
        print( int(current*100/valMax) , "%" )
        verbosePourcent.pourcent = int(current*100/valMax)
    if verbosePourcent.pourcent ==100:
        verbosePourcent.pourcent = 0 # reset le compteur une fois 100% atteint
verbosePourcent.pourcent=0
##########


#####################################################################################
# PROCEDURES/FUNCTIONS:
#####################################################################################
def hoyer(vector):
    '''
    Hoyer's measure of sparsity for a vector
    '''
    sqrt_n = np.sqrt(len(vector))
    return (sqrt_n - np.linalg.norm(vector, 1) / np.linalg.norm(vector)) / (sqrt_n - 1)
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
def treves_rolls(vector):
    '''
    compute modified treve-rolls population sparsennes, formula from (wilmore et all, 2000)
    '''
    denominator = 0
    numerator = 0
    length = len(vector)
    for each in vector:
        numerator += abs(each)
        denominator += (each*each)/length 
    tr=1 - (((numerator/length)*(numerator/length)) /denominator)
    return tr 
#####################################################################################
def reglog(layers, df_metrics,dict_labels):
    '''
    compute a logistic regression for each picture between layer"s metric value (y) and number of layer (x)
    '''
    i = 1.0 
    x = []  
    for each in range(len(layers)):
        x.append(i)
        i += 1   
            
    x = pandas.DataFrame(x) 

    dict_reglog = {}

    for row in df_metrics.iterrows():
        y = []
        j = 0
        for each in list(row)[1]:            
            if j != 0:
                y.append(each)
            j += 1       

        picture = list(row)[0]

        y= pandas.DataFrame(y)
        df = pandas.concat([x,y], axis=1)   
        df.columns = ['x', 'y']
       
        x_stat = sm.add_constant(x)
        
        model = sm.Logit(y, x_stat)
        result = model.fit(disp=0)    
       
        coeff = result.params[0]
        dict_reglog[picture] = coeff        

    df1 = pandas.DataFrame.from_dict(dict_labels, orient='index', columns = ['rate'])
    df2 = pandas.DataFrame.from_dict(dict_reglog, orient='index', columns = ['reglog']) 

    return pandas.concat([df1, df2], axis = 1)     
#####################################################################################
def minmax(df_metrics,dict_labels):
    '''
    compute for each picture the difference between the higher and lower value of layer's metrics 
    '''
    dict_scope = {}

    for row in df_metrics.iterrows():
        y = []
        j = 0
        
        for each in list(row)[1]:
            if j != 0:
                y.append(each)
            j += 1   

        picture = list(row)[0]

        maximum = max(y)
        minimum = min(y)
        diff = maximum - minimum

        dict_scope[picture] = diff     

    df1 = pandas.DataFrame.from_dict(dict_labels, orient='index', columns = ['rate'])
    df2 = pandas.DataFrame.from_dict(dict_scope, orient='index', columns = ['reglog'] ) 
    return pandas.concat([df1, df2], axis = 1)  
#####################################################################################
def gompertzFct (t , N , r , t0 ):
    '''
    Define the Gompertz function model
    '''
    return N * np . exp ( - np . exp ( - r * (t - t0 ))) 
#####################################################################################
def reg_gompertz(x,y, df_gompertz):
    '''
    Compute a regression on the Gompertz function. 
    Function for the moment not functional. 
    '''
    I_t = y [ x :]
    t = np.arange (len( I_t ))

    model = gompertzFct
    guess = (100000. , .1 , 50.)

    parameters , variances = opt . curve_fit ( model , t , I_t , p0 = guess )

    G_t = model (t , * parameters )

    print ( np . sqrt ( np . mean (( I_t - G_t )**2)))
#####################################################################################
def histplot_metrics(layers, df_metrics, bdd, weight, metric, log_path,k):
    '''
    Plot a histogram of the distribution of metrics for all images 
    regardless of the layer in which they were calculated
    '''    
    y = []    
    for layer in layers:
        if layer[0:5] == 'input':
            layer = 'input' + '_' + str(k)         
        y = list(itertools.chain(y, list(df_metrics[layer])))
    title = 'distrib_'+ bdd +'_'+ weight +'_'+ metric   
    plt.hist(y, bins = 40)        
    plt.title(title, fontsize=10)                 
    plt.savefig(log_path +'_'+ bdd +'_'+ weight +'_'+ metric +'.png')
    plt.clf()
#####################################################################################
def compress_metric(df_metrics, metric):
    '''
    If metrics values aren't between 0 and 1, like kurtosis or L0 norm, change theme like this
    to let them availaible for logistic regression. 1 will be th highest value of the metric, 0 the lowest.
    '''
    vmax = 0
    vmin = 0

    #suppression temporaire de la note
    rates = df_metrics.pop('rate')     

    #récupération de la valeur minimale
    j = 0
    for row in df_metrics.index:        
        i = 0
        for index_row, row in df_metrics.iterrows():                     
            for column in row:           
                if i != 0:                                
   
                    if column < vmin:
                        vmin = column                 
            i += 1
        j += 1
    
    #transformation des valeurs pour qu'elles soient strictement positives    
    df_metrics = df_metrics.applymap(lambda x: x + abs(vmin)) 

    #récupération de la valeur maximale
    j = 0
    for row in df_metrics.index:        
        i = 0
        for index_row, row in df_metrics.iterrows():                     
            for column in row:           
                if i != 0:                                
                    if column > vmax:
                        vmax = column                   
            i += 1
        j += 1   
     
    #transformation des valeurs pour qu'elles soient entre 0 et 1    
    df_metrics = df_metrics.applymap(lambda x: x/vmax) 

    #ajout de la note
    df_metrics.insert(0, 'rate', rates, allow_duplicates=False)    

    return df_metrics
#####################################################################################
def inflexion_points(df_metrics,dict_labels):
    '''
    from here: https://stackoverflow.com/questions/62537703/how-to-find-inflection-point-in-python
    '''
    dict_inflexions = {}

    for row in df_metrics.iterrows():
        y = []
        j = 0
        
        for each in list(row)[1]:
            if j != 0:
                y.append(each)
            j += 1   

        picture = list(row)[0]


        smooth = gaussian_filter1d(y, 3)

        # compute second derivative
        smooth_d2 = np.gradient(np.gradient(smooth))

        smooth_d1 = np.gradient(smooth)
        
        y_d1 = np.gradient(y)

        # find switching points
        infls = np.where(np.diff(np.sign(smooth_d2)))[0]

        coeff = y_d1[max(infls)]      

          

        dict_inflexions[picture] = coeff     


    df1 = pandas.DataFrame.from_dict(dict_labels, orient='index', columns = ['rate'])
    df2 = pandas.DataFrame.from_dict(dict_inflexions, orient='index', columns = ['reglog'] ) 
    return pandas.concat([df1, df2], axis = 1)  

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


######################################
def getVarienceRatio(pca, bdd, layer, pathData = "../../"):
   
    variance = pca.explained_variance_ratio_ #calculate variance ratios
    var=np.cumsum(pca.explained_variance_ratio_) * 100
    print( var) #cumulative sum of variance explained with [n] features
    df = pandas.DataFrame(variance).transpose()
    df2 = pandas.DataFrame(var).transpose()
    os.makedirs(pathData+"_variance", exist_ok=True)
    df.to_csv(pathData+"_variance"+"/"+"variance"+layer+".csv")
    df2.to_csv(pathData+"_variance"+"/"+"varianceCumule_"+layer+".csv")



def Acp_densiteProba(dict_metrics, pc, bdd, layer):
    '''
    A PCA with activations of each bloc as features
    '''
    df_metrics = pandas.DataFrame.from_dict(dict_metrics)     
      
    tic = time.perf_counter()    





