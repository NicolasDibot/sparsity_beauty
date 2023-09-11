#!/usr/bin/env python
#####################################################################################
# DESCRIPTION:
#####################################################################################
#Main program of the PCA calculation on the activations of the intermediate layers of VGG16
#Output: image coordinates for each layer, for the PCA components representing 80% of the explained variance
#The goal is then to create a model under R (for example an e ridge regression in leave one out cross validation) with these values as explanatory variables and the beauty/attractiveness as variable to explain
# Loop on the combinatorics of the parameters (databases, weights, model etc)

#choice of parameters below
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

pathData = '../../'
if len(sys.argv) >1:
    if sys.argv[1]== 'mesoLR':
        sys.path.insert(1,'/home/tieos/work_swp-gpu/melvin/thesis/code/functions')
        pathData = '/home/tieos/work_swp-gpu/melvin/thesis/'
    if sys.argv[1]== 'mesoLR-3T':
        sys.path.insert(1,'/home/tieos/work_cefe_swp-smp/melvin/thesis/code/functions')
        pathData = '/home/tieos/work_cefe_swp-smp/melvin/thesis/'
    elif sys.argv[1] == 'sonia':
        pathData =  '/media/sonia/DATA/data_nico/'
print("path: ", sys.path)
print("\n Current working directory: {0}".format(os.getcwd()))
import sparsenesslib.high_level as hl

#####################################################################################
#SETTINGS:
#####################################################################################
PIL.Image.MAX_IMAGE_PIXELS = 30001515195151997
478940                             

model_name = 'VGG16'  # 'vgg16, resnet (...)'
#weights = 'vggface' #'imagenet','vggface'
list_weights = ['imagenet'] #['vggface','imagenet','vggplace']
#computer = 'LINUX-ES03' #no need to change that unless it's sonia's pc, that infamous thing; in which case, put 'sonia' in parameter.
freqmod = 100 #frequency of prints, if 5: print for 1/5 images
AllPC=[]

list_bdd = ""
method = ""
if len(sys.argv) >3:
    list_bdd = sys.argv[2].split(",")
    method = sys.argv[3]
else:
    #'CFD','SCUT-FBP','MART','JEN','SMALLTEST','BIGTEST'
    list_bdd = [ 'CFD_AF','CFD_F'] #"['CFD','MART','JEN','SCUT-FBP','SMALLTEST','BIGTEST']"
    list_bdd = ['MART']
    list_bdd = ['CFD_ALL']
    list_bdd = ['SMALLTEST']

    #method = "average"#_FeatureMap"
    method = "FeatureMap"
    #method = "max"#_FeatureMap"
    method = "pca"


#####################################################################################
#CODE
#####################################################################################
list_metrics = ['acp']
k = 1

#pour plots les PC



l = len(list_bdd)*len(list_weights)*len(list_metrics)
for bdd in list_bdd:
    for weight in list_weights:
        for metric in list_metrics:
            loadModele = ""
            #loadModele = "results/SMALLTEST/FeatureMap"


            print('###########################--COMPUTATION--#################################_STEP: ',k,'/',l,'  ',bdd,', ',weight,', ',metric)
            #hl.extract_pc_acp_block(bdd,weight,metric, model_name, pathData, freqmod,k)
            #k += 1
            #computation='featureMap'
            if method == "pca":
                method='flatten'
            hl.preprocess_ACP(bdd,weight,metric, model_name, pathData, freqmod,k, method,saveModele = False,loadModele = loadModele)
            k += 1

    
#####################################################################################

