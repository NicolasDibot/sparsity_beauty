#!/usr/bin/env python
#####################################################################################
# DESCRIPTION:
#####################################################################################
#high level functions that organize the sequence of the other functions of the module, they would probably be difficult to reuse for another project

#1. compute_sparseness_metrics_activations: compute metrics of the layers given in the list *layers* of the images contained in the directory *path*
    #by one of those 3 modes: flatten channel or filter (cf activations_structures subpackage) and store them in the dictionary *dict_output*.

#2. write_file: Writes the results of the performed analyses and their metadata in a structured csv file with 
    # a header line, 
    # results (one line per layer), 
    # a line with some '###', 
    # metadata

#3. layers_analysis: something like a main, but in a function (with all previous function),also, load paths, models/weights parameters and write log file

#####################################################################################
# LIBRAIRIES:
#####################################################################################
#public librairies
import time
import os
from tensorflow.keras.preprocessing.image import load_img 
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import VGG16
#from keras_vggface.vggface import VGGFace
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
import sys
import statistics as st
from scipy import stats
from datetime import date
from more_itertools import chunked
import pandas
import matplotlib.pyplot as plt
import numpy as np
import json
import re
import csv
#personnal librairies
sys.path.insert(1,'../../code/functions')
import sparsenesslib.keract as keract
import sparsenesslib.metrics as metrics
import sparsenesslib.metrics_melvin as metrics_melvin
import sparsenesslib.plots as plots

import sparsenesslib.sparsenessmod as spm
import sparsenesslib.activations_structures as acst

#####################################################################################
# PROCEDURES/FUNCTIONS:
#####################################################################################

def getPaths(bdd, pathData):
        #path d'enregistrement des résultats

    
    if pathData == 'LINUX-ES03':
        pathData = '../../'

    if bdd in ['CFD','JEN','SCUT-FBP','MART','CFD_1','CFD_AF','CFD_F','CFD_WM']:
        labels_path =pathData+'data/redesigned/'+bdd+'/labels_'+bdd+'.csv'
        images_path =pathData+'data/redesigned/'+bdd+'/images'
        log_path =pathData+'results/'+bdd+'/log_'
    elif bdd in ['CFD_ALL']:
        labels_path =pathData+'data/redesigned/CFD/labels_CFD.csv'
        images_path =pathData+'data/redesigned/CFD/images'
        log_path =pathData+'results/CFD/log_'
    elif bdd == 'SMALLTEST':
        labels_path =pathData+'data/redesigned/small_test/labels_test.csv'
        images_path =pathData+'data/redesigned/small_test/images'
        log_path =pathData+'results/smalltest/log_'            
    elif bdd == 'BIGTEST':
        labels_path =pathData+'data/redesigned/big_test/labels_bigtest.csv'
        images_path =pathData+'data/redesigned/big_test/images'
        log_path =pathData+'results/bigtest/log_'  
    elif bdd == 'Fairface':
        labels_path =pathData+'data/redesigned/Fairface/fairface_label_train.csv'
        images_path =pathData+'data/redesigned/Fairface/'
        log_path =pathData+'results/Fairface/log_'  
    return labels_path, images_path, log_path
    
#####################################################################################

def configModel(model_name, weight):
    if model_name == 'VGG16':
        if weight == 'imagenet':
            model = VGG16(weights = 'imagenet')
            layers = ['input_1','block1_conv1','block1_conv2','block1_pool','block2_conv1', 'block2_conv2','block2_pool',
            'block3_conv1','block3_conv2','block3_conv3','block3_pool','block4_conv1','block4_conv2','block4_conv3',
            'block4_pool', 'block5_conv1','block5_conv2','block5_conv3','block5_pool','flatten','fc1', 'fc2']
            flatten_layers = ['fc1','fc2','flatten']
        elif weight == 'vggface':
            model = VGGFace(model = 'vgg16', weights = 'vggface')
            layers = ['input_1','conv1_1','conv1_2','pool1','conv2_1','conv2_2','pool2','conv3_1','conv3_2','conv3_3',
            'pool3','conv4_1','conv4_2','conv4_3','pool4','conv5_1','conv5_2','conv5_3','pool5','flatten',
            'fc6/relu','fc7/relu']
            flatten_layers = ['flatten','fc6','fc6/relu','fc7','fc7/relu','fc8','fc8/softmax']
        elif weight == 'vggplaces':
            model = places.VGG16_Places365(weights='places')
            layers = ['input_1','block1_conv1','block1_conv2','block1_pool','block2_conv1', 'block2_conv2','block2_pool',
            'block3_conv1','block3_conv2','block3_conv3','block3_pool','block4_conv1','block4_conv2','block4_conv3',
            'block4_pool', 'block5_conv1','block5_conv2','block5_conv3','block5_pool','flatten','fc1', 'fc2']
            flatten_layers = ['fc1','fc2','flatten']
    elif model_name == 'resnet50':
        if weight == 'imagenet': 
            print('error, model not configured')
        elif weight == 'vggfaces':
            print('error, model not configured')  
    return model, layers, flatten_layers

#####################################################################################

def preprocess_Image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    image = img_to_array(img)      
    img = image.reshape(
        (1, image.shape[0], image.shape[1], image.shape[2]))   
    image = preprocess_input(img)
    return image

#####################################################################################

def compute_sparseness_metrics_activations(model, flatten_layers, path, dict_output, layers, computation, formula, freqmod,k):
    '''
    compute metrics of the layers given in the list *layers*
    of the images contained in the directory *path*
    by one of those 3 modes: flatten channel or filter (cf activations_structures subpackage)
    and store them in the dictionary *dict_output*.
    '''
    imgs = [f for f in os.listdir(path)]    
    
    for i, each in enumerate(imgs,start=1):

        if i%freqmod == 0:
            print('###### picture n°',i,'/',len(imgs),'for ',formula, ', ', computation)
      
        img_path = path + "/" + each
        image = preprocess_Image(img_path)

        activations = keract.get_activations(model, image)
        activations_dict = {}
        acst.compute_activations(layers, flatten_layers, computation, activations, activations_dict,formula,k)
        dict_output[each] = activations_dict
        
#####################################################################################

def getActivations_for_all_image(model,path, imgs, computation, formula, freqmod):
    '''
    Returns a dictionary by image of activations     
    '''  
    #print("the path is: ", path)
    imageActivation = {}
    #imgs = [f for f in os.listdir(path)]
    for i, each in enumerate(imgs,  start=1):
        if i%freqmod == 0:         
            print('###### picture n°',i,'/',len(imgs),'for ',formula, ', ', computation)

        
        img_path = path + "/" + each
        image = preprocess_Image(img_path)
        
        # récupération des activations
        activations = keract.get_activations(model, image)
        
        imageActivation[each] = activations
    return imageActivation

#####################################################################################

def get_activation_by_layer(activations,imgList,dict_output,computation, formula, k, layer):
    """
    from the dictionary of all activations, extract the selected layer
    """
    #for i, each in enumerate([f for f in os.listdir(path)],  start=1) :
    for i, each in enumerate(imgList,  start=1) :
        activations_dict = {}
        
        if computation == 'flatten' or layer in ['fc1','fc2','flatten']:
            if formula in ['mean', 'max']:
                formula = "acp"
            acst.compute_flatten(activations[each], activations_dict, layer, formula,k)
        elif computation == 'featureMap':
            acst.compute_flatten_byCarte(activations[each], activations_dict, layer, formula,k)
        else:
            print('ERROR: Computation setting isnt flatten or featureMap')
            return -1


        dict_output[each] = activations_dict

#####################################################################################

def parse_activations_by_layer(model,path, dict_output, layer, computation, formula, freqmod,k):
    
    '''
    a function that for the layer and only the layer, stores the activations of all the images
    it returns the array of activations at the chosen layer 
    '''
    
    imgs = [f for f in os.listdir(path)] 
  
    
    for i, each in enumerate(imgs,start=1):
        if i%freqmod == 0:         
            print('###### picture n°',i,'/',len(imgs),'for ',formula, ', ', computation)
       
        
        img_path = path + "/" + each
        image = preprocess_Image(img_path)
        
        # récupération des activations
        activations = keract.get_activations(model, image)
        
        activations_dict = {}
        
        acst.compute_flatten(activations, activations_dict, layer, formula,k)
        
        dict_output[each] = activations_dict
#####################################################################################
def parse_activations_by_filter(model,path, list_output, layer, computation, formula, freqmod,k):
    
    '''
    A function that for the layer and only the layer, stores the activations of all the images, by filter.  
    It returns in list_output a list (of size n = number of filters) of dictionaries 
    (of size n = number of images) of the array of activations per filter at the chosen layer
    '''
    
    imgs = [f for f in os.listdir(path)] 
      

    #to get the number of activations, test on any image
    img_path = path + "/" + imgs[1]
    image = preprocess_Image(img_path)


    activations = keract.get_activations(model, image)
    print(layer)    
    [print(k, '->', v.shape, '- Numpy array') for (k, v) in activations.items()]


    for i, each in enumerate(imgs,start=1):
        if i%freqmod == 0:         
            print('###### picture n°',i,'/',len(imgs),'for ',formula, ', ', computation)
       
        
        img_path = path + "/" + each
        image = preprocess_Image(img_path)
        
        # activation recovery
        activations = keract.get_activations(model, image)
        
        activations_dict = {}
        
        acst.compute_flatten(activations, activations_dict, layer, formula,k)  
        
        dict_output[each] = activations_dict
        
#####################################################################################
def write_file(log_path, bdd, weight, metric, df_metrics, df_reglog, df_scope, df_inflexions, layers, k):    
    '''
    Writes the results of the performed analyses and their metadata in a structured csv file with 
    - a header line, 
    - results (one line per layer), 
    - a line with some '###', 
    - metadata
    '''

    today = date.today()
    today = str(today)
    for i in range(2,31):
        df_metrics = df_metrics.rename(columns = {'input_'+i: 'input_1'})

    with open(log_path +'_'+bdd+'_'+weight+'_'+metric+'_'+today+'_ANALYSE'+'.csv',"w") as file:            
        #HEADER
        file.write('layer'+';'+'mean_'+str(metric)+';'+'sd_'+str(metric)+';'+'corr_beauty_VS_'+'metric'+';'+'pvalue'+';'+'\n')
        #VALUES for each layer
        for layer in layers:

            '''
            if layer[0:5] == 'input':
                layer = 'input' + '_' + str(k)'''
            file.write(layer+';')            
            #mean
            l1 = list(df_metrics[layer])
            file.write(str(st.mean(l1))+';')               
            #standard deviation
            l1 = list(df_metrics[layer])
            file.write(str(st.stdev(l1))+';')            
            #correlation with beauty
            l1 = list(df_metrics[layer])
            l2 = list(df_metrics['rate']) 
            reg = linregress(l1,l2)
            r = str(reg.rvalue)         
            file.write(r +';')             
            #pvalue
            pvalue = str(reg.pvalue) 
            file.write(pvalue+';'+'\n')   
        
        #METADATA
        file.write('##############'+'\n')        
        file.write('bdd;'+ bdd + '\n')        
        file.write('weights;'+ weight + '\n')         
        file.write('metric;'+ metric + '\n')            
        file.write("date:;"+today+'\n')
        #correlation with scope
        l1 = list(df_scope['reglog'])        
        l2 = list(df_scope['rate'])
        reg = linregress(l1,l2)
        coeff = str(reg.rvalue) 
        pvalue = str(reg.pvalue)
        file.write('coeff_scope: ;'+coeff+';pvalue:'+pvalue +'\n') 
        #correlation with coeff of logistic regression
        l1 = list(df_reglog['reglog'])        
        l2 = list(df_reglog['rate'])
        reg = linregress(l1,l2)
        coeff = str(reg.rvalue)
        pvalue = str(reg.pvalue)    
        file.write('coeff_reglog: ;'+coeff+';pvalue:'+pvalue +'\n')  
        #correlation with each inflexions points        
        l1 = list(df_inflexions['reglog'])        
        l2 = list(df_inflexions['rate'])
        reg = linregress(l1,l2)
        coeff = str(reg.rvalue) 
        pvalue = str(reg.pvalue)       
        file.write('coeff_slope_inflexion: ;'+coeff+';pvalue:'+pvalue +'\n') 
        
#####################################################################################
def extract_metrics(bdd,weight,metric, model_name, computer, freqmod,k = 1):
    '''
    something like a main, but in a function (with all previous function)
    ,also, load paths, models/weights parameters and write log file

    *k:index of the loop, default is 1*'''

    t0 = time.time()

    
    labels_path, images_path, log_path = getPaths(bdd, computer)
    model, layers, flatten_layers =configModel(model_name, weight)

    dict_compute_metric = {}    
    dict_labels = {}

    if metric == 'L0':
        compute_sparseness_metrics_activations(model,flatten_layers, images_path,dict_compute_metric, layers, 'flatten', metric, freqmod,k)
    if metric == 'kurtosis':
        compute_sparseness_metrics_activations(model,flatten_layers, images_path,dict_compute_metric, layers, 'flatten', metric, freqmod,k)
    if metric == 'L1':
        compute_sparseness_metrics_activations(model,flatten_layers, images_path,dict_compute_metric, layers, 'flatten', metric, freqmod,k)
    if metric == 'hoyer':
        compute_sparseness_metrics_activations(model,flatten_layers, images_path,dict_compute_metric, layers, 'flatten', metric, freqmod,k)
    if metric == 'gini_flatten':
        compute_sparseness_metrics_activations(model,flatten_layers, images_path,dict_compute_metric, layers, 'flatten', 'gini', freqmod, k)
    if metric == 'gini_channel':
        compute_sparseness_metrics_activations(model,flatten_layers, images_path,dict_compute_metric, layers, 'channel', 'gini', freqmod, k)
    if metric == 'gini_filter':
        compute_sparseness_metrics_activations(model,flatten_layers, images_path,dict_compute_metric, layers, 'filter', 'gini', freqmod, k)
    if metric == 'mean':
        compute_sparseness_metrics_activations(model,flatten_layers, images_path,dict_compute_metric, layers, 'flatten', metric, freqmod, k)
    if metric == 'acp':
        compute_sparseness_metrics_activations(model,flatten_layers, images_path,dict_compute_metric, layers, 'flatten', metric, freqmod, k)

    
    spm.parse_rates(labels_path, dict_labels)
    
    today = date.today()
    today = str(today)

    df_metrics = spm.create_dataframe(dict_labels, dict_compute_metric) 
    today = date.today()
    today = str(today)
    df_metrics.to_json(path_or_buf = log_path+'_'+bdd+'_'+weight+'_'+metric+'_'+'_BRUTMETRICS'+'.csv')
#####################################################################################

def preprocess_ACP(bdd,weight,metric, model_name, computer, freqmod,k = 1,computation = 'flatten',saveModele = False,loadModele=""):
    '''
    Met en place les composants pour l'execution de l'ACP.   
    adapte les chemins d'entrée et sortie.
    prétraite certaines bases de donnée (CFD_ALL, fairface)
    '''

    def extract_pc_acp():#bdd, layers, computation, freqmod, model, images_path,imglist, k, loadModele, metric, path, saveModele):
        '''
        something like a main, but in a function (with all previous function)
        ,also, load paths, models/weights parameters and write log file

        *k:index of the loop, default is 1*

        Version for compute pca (loop on layers before loop on pictures)    
        '''
    
        print("longueur imglist: ", len(imglist))
        activations = getActivations_for_all_image(model,images_path,imglist,computation, metric, freqmod)
    
    
        #nbComp = pandas.DataFrame()
        for layer in layers:
    
        
            print('##### current layer is: ', layer)
            #une fonction qui pour la couche et seulement la couche, stocke les activations de toutes les images
            #elle retourne l'array des activations à la couche choisie
            dict_activations = {}
            get_activation_by_layer(activations,imglist,dict_activations,computation, metric, k, layer)
        
            #parse_activations_by_layer(model,images_path,dict_activations, layer, 'flatten', metric, freqmod, k)
        
            pc = []
            #une fonction qui fait une acp la dessus, qui prends en entrée la liste pc vide et l'array des activations,
            #et enregistre les coordonnées des individus pour chaque composante dans un csv dans results/bdd/pca
        
    
            if computation == 'flatten' or layer in ['fc1','fc2','flatten']:
                if loadModele!="":
                    comp = metrics.acp_layers_loadModele(dict_activations, pc, bdd, layer, path,modelePath = loadModele)
                else:
                    comp = metrics.acp_layers(dict_activations, pc, bdd, layer, path,saveModele = saveModele)
            elif computation == 'featureMap':
                if loadModele!="":
                    comp = metrics.acp_layers_featureMap_loadModele(dict_activations, pc, bdd, layer, path,modelePath = loadModele)
                else:
                    comp = metrics.acp_layers_featureMap(dict_activations, pc, bdd, layer, path, saveModele = saveModele)
    
    allCFD = False
    if bdd == "CFD_ALL":
        allCFD = True
        bdd = "CFD"
   
    t0 = time.time()

    if computer == 'LINUX-ES03':
        computer = '../../'

    labels_path, images_path, log_path = getPaths(bdd, computer)
    model, layers, flatten_layers =configModel(model_name, weight)
    
    #sur le mesoLR, le chemin d'écriture et de lecture est différent
    if computer == '/home/tieos/work_cefe_swp-smp/melvin/thesis/':  #lecture
            computer = '/lustre/tieos/work_cefe_swp-smp/melvin/thesis/' #ecriture
    
    ## indique le chemin a charger si on charge le modèle        
    if loadModele !="":
        loadModele = computer+loadModele
    #adapte le chemin suivant la methode 
    if computation == 'flatten':
        path= computer+"results"+"/"+bdd+"/pca"
    elif computation == 'featureMap': 
        path= computer+"results"+"/"+bdd+"/FeatureMap"


    #dict_compute_pc = {}   #un dictionnaire qui par couche, a ses composantes principales (et les coorodnnées de chaque image pour chaque composante)
    dict_labels = {}
    print("path :", computer)

   ####### lance l'ACP sur chaque sous ensemble de CFD
    if allCFD == True:
        combinaison = getAllGenreEthnieCFD(labels_path)#, exception= {'ethnie' : ["M","I"]})

        for key in combinaison.keys():
            if key == "":
                bdd = "CFD"
            else:
                bdd = "CFD_"+key
            imglist = combinaison[key]
            #adapte le chemin suivant la methode 
            if computation == 'flatten':
                path= computer+"results"+"/"+bdd+"/pca"
            elif computation == 'featureMap': 
                path= computer+"results"+"/"+bdd+"/FeatureMap"
            extract_pc_acp()#bdd,layers, computation, freqmod,  model,images_path, imglist, k, loadModele, metric, path, saveModele)
    #######   
    else:
        if bdd == "Fairface":
            #filt = {'ethnie' : "Asian", 'genre' : "Female"}
            filt = {'ethnie' : "White", 'genre' : "Male"}
            imglist = parserFairface(labels_path,filt)
            #imglist = parserFairface(labels_path)
            for key, item in filt.items():
                if bdd == "Fairface":
                    bdd = bdd+"_"
                bdd = bdd+item[0]
            print("BDD: ", bdd,"\n\n")
        else:
            imglist = [f for f in os.listdir(images_path)]
    

        extract_pc_acp()#bdd, layers, computation, freqmod,  model,images_path,imglist, k, loadModele, metric, path, saveModele)
        
    if '/lustre/tieos/work_cefe_swp-smp/melvin/thesis/' in path:
        path = '/home/tieos/work_cefe_swp-smp/melvin/thesis/'
    
    spm.parse_rates(labels_path, dict_labels)
    
    today = date.today()
    today = str(today)

   
#####################################################################################
def analyse_metrics(model_name, computer, bdd, weight, metric,k):
  
    model, layers, flatten_layers =configModel(model_name, weight)

    labels_path, images_path, log_path = getPaths(bdd, computer)

    
    dict_labels = {}
    spm.parse_rates(labels_path, dict_labels)
    
    data = json.load(open(log_path+'_'+bdd+'_'+weight+'_'+metric+'_'+'_BRUTMETRICS'+'.csv', "r"))    
    df_metrics = pandas.DataFrame.from_dict(data)    
      
    if metric in ['kurtosis','L0','mean']:              
        df_metrics = metrics.compress_metric(df_metrics, metric)  
        
    df_reglog = metrics.reglog(layers, df_metrics, dict_labels) 
        
    df_scope = metrics.minmax(df_metrics,dict_labels)    
       
    
    df_inflexions = metrics.inflexion_points(df_metrics,dict_labels)
    df_inflexions.to_json(path_or_buf = log_path+'_'+bdd+'_'+weight+'_'+metric+'_'+'_inflexions'+'.csv')
    
    write_file(log_path, bdd, weight, metric, df_metrics, df_reglog, df_scope, df_inflexions ,layers, k)    
#####################################################################################
def getAllFile(path, formatOrdre = []):
    """! browse all files in the path directory in the order indicated in formatOrdre
    @param path chemin du répertoire
    @param formatOrdre [optional] allows you to browse the directory in a specific order:
        syntax: formatOrdre[  prefixe, TabName[], sufixe]

    @return list of file names
    """
    files = []
    if len(formatOrdre)==0: 
        files = [f for f in os.listdir(path)]    
    else: 
        files = [formatOrdre[0]+f+formatOrdre[2] for f in formatOrdre[1]]
    return files
