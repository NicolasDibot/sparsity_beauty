#####################################################################################
# 1. DESCRIPTION:
#####################################################################################

#####################################################################################
# 2. LIBRAIRIES:
#####################################################################################
library("rjson")
library("readr")
library("purrr")
library("tidyr")
library("tibble")
library("plyr")
library("corrplot")
library("dplyr")
library("caret")
library("jtools")
library("broom.mixed")
library("glmnet")
library("tidyverse")
setwd() #working directory

kfold_pca <- function(bdd, weight, metric, layer, regularization, print_number) {
  
    print('######')
    print(layer)
    ######################
    # DATA MANAGEMENT
    ######################
    labels_path = paste('../../data/redesigned/',bdd,'/labels_',bdd,'.csv', sep="")
    log_path =paste('../../results/',bdd,'/pca/', sep="")
    log_path_rate =paste('../../results/',bdd,'/log_', sep="")
    
    df_pc = read_csv(file = paste(log_path,"pca_values_",layer,".csv", sep =""), show_col_types = FALSE)
    df_pc = df_pc[,-1]
    
    matrix_metrics <- do.call(cbind, fromJSON(file = paste(log_path_rate,'_',bdd,'_',weight,'_',metric,'_','_BRUTMETRICS','.csv',sep=""),simplify = FALSE))
    df_metrics <- as.data.frame(matrix_metrics, optional = TRUE)
    df_metrics = sapply(df_metrics, as.numeric)
    df_metrics <- as.data.frame(df_metrics)
    df = cbind(df_metrics$rate, df_pc)
    df <- plyr::rename(df, c("df_metrics$rate" = "rate"))
    df$sp = df_metrics[,layer] 
    
    ###############################
    #MODEL
    ###############################
    ctrl = trainControl(method = "repeatedcv", number = 10) #10-fold cv
    lambdas = 10^seq(2,-4,by=-0.1)
    model1 = train( rate ~ ., data = df ,method = "glmnet", tuneGrid = expand.grid(alpha = 0, lambda = lambdas),preProc = c("center", "scale"),trControl = ctrl, metric = "Rsquared")
    
    alpha = model1$results$alpha[1]
    lambda = model1$results$lambda[1]
    r_squared = model1$results$Rsquared[1]
    
    list = list('r_squared' = r_squared, 'AIC'= 1, 'BIC'= 1 )
    
    return(list)

}
#####################################################################################
# 4. PARAMETERS:
#####################################################################################
bdd <- c('JEN')
weight <- c('imagenet')
metric <- c('gini_flatten')
layers <-   c('block1_conv1','block1_conv2',
             'block2_conv1','block2_conv2',
             'block3_conv1','block3_conv2','block3_conv3',
             'block4_conv1','block4_conv2','block4_conv3',
             'block5_conv1','block5_conv2','block5_conv3',
             'fc1','fc2')
regularization <- 'ridge' #0 for ridge, 1 for lasso
print_number = 200

set.seed(238)

######################################################################################
# 5. MAIN:
######################################################################################
R_squareds = c()


for (layer in layers){
  results = kfold_pca(bdd, weight, metric, layer, regularization, print_number)
  R_squareds = c(R_squareds, results$r_squared)

}
names_layers <-c('1_1','1_2',
                 '2_1','2_2',
                 '3_1','3_2','3_3',
                 '4_1','4_2','4_3',
                 '5_1','5_2','5_3',
                 'fc1','fc2')


df = data.frame(names_layers, bdd)
write.csv(df,paste("fig.csv"),row.names = TRUE)



