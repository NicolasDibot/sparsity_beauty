# README

This repository provides the code used for the article
**Sparsity in an artificial neural network predicts beauty: towards a model of processing-based aesthetics**
by Dibot et al. 

Warning: the data necessary for the proper functioning of these scripts are not provided. It is necessary to contact the respective authors of the databases *(CFD, SCUT-FBP, MART, JEN)* mentioned in the above article to obtain them. 

1. The script **code/pre_trained_models/main_models.py** allows to extract the activations of a neural network for all the images of a database
2. The script **code/pre_trained_models/main_metrics.py** allows to compute the Gini index on the previously computed activations
3. The script **code/pre_trained_models/main_pca.py** allows to perform PCA on previously calculated activations

4. The R scripts contained in **code/models.R** allow to realize the statistical tests whose results are presented in the article. They are named after the figure or table in the article whose results they are used to obtain. 

5. The functions contained in **code/functions** are those used in the scripts contained in **code/pre_trained_models** but are stored in several files separated according to their role for greater readability and clarity of the code.

6. The **environment.yml** file is used to install the necessary python libraries in the form of a conda environment. 


Don't hesitate to ask "issue" questions, I'll answer them and help other people. 