#!/usr/bin/env python
#####################################################################################
# DESCRIPTION:
#####################################################################################
#Main program for calculating sparsity correlated metrics on neural network middle layers. 
#Loop on the combinatorics of the parameters (databases, weights, model etc)
#Choice of these parameters below. 

#####################################################################################
# LIBRAIRIES:
#####################################################################################
#public librairies
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import PIL
import sys
#personnal librairies
sys.path.insert(1,'../../code/functions')
import sparsenesslib.high_level as hl
#####################################################################################
#SETTINGS:
#####################################################################################
PIL.Image.MAX_IMAGE_PIXELS = 30001515195151997
478940                             
#'CFD','SCUT-FBP','MART','JEN','SMALLTEST','BIGTEST'
list_bdd = ['CFD','MART','JEN','SCUT-FBP','SMALLTEST','BIGTEST'] #"['CFD','MART','JEN','SCUT-FBP','SMALLTEST','BIGTEST']"
list_bdd = ['SCUT-FBP']
model_name = 'VGG16'  # 'vgg16, resnet (...)'
#weights = 'vggface' #'imagenet','vggface'
list_weights = ['imagenet'] #['vggface','imagenet','vggplace']
list_metrics = ['gini_flatten'] #['L0','L1','gini_flatten','gini_channel','gini_filter','kurtosis']
computer = 'LINUX-ES03' 
freqmod = 250 #frequency of prints, if 5: print for 1/5 images
#####################################################################################
#CODE
#####################################################################################
k = 1
l = len(list_bdd)*len(list_weights)*len(list_metrics)
for bdd in list_bdd:    
    for weight in list_weights:
        for metric in list_metrics:
            print('###########################--COMPUTATION--#################################_STEP: ',k,'/',l,'  ',bdd,', ',weight,', ',metric)
            hl.extract_metrics(bdd,weight,metric, model_name, computer, freqmod,k)            
            k += 1
#####################################################################################
