#!/usr/bin/env python
#####################################################################################
# DESCRIPTION:
#####################################################################################
#Main program to analyze the metrics computed with main_metrics.py to detect correlations with beauty/attractiveness scores
#and build a predictive model of beauty based on these metrics and their dynamics. 

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
list_bdd = ['CFD'] #"['CFD','MART','JEN','SCUT-FBP','SMALLTEST','BIGTEST']"
model_name = 'VGG16'  # 'vgg16, resnet (...)'
#weights = 'vggface' #'imagenet','vggface'
list_weights = ['imagenet'] #['vggface','imagenet','vggplace']
list_metrics = ['hoyer'] #['L0','L1','gini_flatten','gini_channel','gini_filter','kurtosis']
computer = 'LINUX-ES03' #no need to change that unless it's sonia's pc, that infamous thing; in which case, put 'sonia' in parameter.
freqmod = 10 #frequency of prints, if 5: print for 1/5 images
#####################################################################################
#CODE
#####################################################################################
k = 1
l = len(list_bdd)*len(list_weights)*len(list_metrics)
for bdd in list_bdd:    
    for weight in list_weights:
        for metric in list_metrics:
            print('######################--MODELISATION--######################################_STEP: ',k,'/',l,'  ',bdd,', ',weight,', ',metric)
            hl.analyse_metrics(model_name, computer, bdd, weight, metric,k)            
            k += 1
#####################################################################################

 
