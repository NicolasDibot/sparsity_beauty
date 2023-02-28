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
setwd("") #write here the working directory
#####################################################################################
# 3. Functions
#####################################################################################

get_3comp_pca <- function(bdd, weight, metric, layer, print_number) {
  
    print('######')
    print(layer)
    ######################
    # 3.2.1 DATA MANAGEMENT
    ######################
    labels_path = paste('../../data/redesigned/',bdd,'/labels_',bdd,'.csv', sep="")
    log_path =paste('../../results/',bdd,'/average/', sep="")
    
    sp_path =paste('../../results/',bdd,'/log_', sep="")
    matrix_metrics <- do.call(cbind, fromJSON(file = paste(sp_path,'_',bdd,'_',weight,'_',metric,'_','_BRUTMETRICS','.csv',sep=""),simplify = FALSE))
    colnames(matrix_metrics)[2] <- 'input_1'
    df_metrics <- as.data.frame(matrix_metrics, optional = TRUE)
    df_metrics = sapply(df_metrics, as.numeric)
    df_metrics <- as.data.frame(df_metrics)

    df_pc = read_csv(file = paste(log_path,"average_values_",layer,".csv", sep =""), show_col_types = FALSE)
    df_pc = df_pc[,-1]
    df_pc <- as.data.frame(df_pc)
    
    return(cbind(df_pc[,1:3],df_metrics[layer]))
}
#####################################################################################
# 4. PARAMETERS:
#####################################################################################
bdd <- c('MART')
weight <- c('imagenet')
metric <- c('gini_flatten')
layers <-   c('block1_conv1','block1_conv2',
             'block2_conv1','block2_conv2',
             'block3_conv1','block3_conv2','block3_conv3',
             'block4_conv1','block4_conv2','block4_conv3',
             'block5_conv1','block5_conv2','block5_conv3',
             'fc1','fc2')
regularization <- 'ridge' #0 for ridge, 1 for lasso

set.seed(023)

######################################################################################
# 5. MAIN:
######################################################################################

log_path_rate =paste('../../results/',bdd,'/log_', sep="")
matrix_metrics <- do.call(cbind, fromJSON(file = paste(log_path_rate,'_',bdd,'_',weight,'_',metric,'_','_BRUTMETRICS','.csv',sep=""),simplify = FALSE))
df_metrics <- as.data.frame(matrix_metrics, optional = TRUE)
df_metrics = sapply(df_metrics, as.numeric)
df_metrics <- as.data.frame(df_metrics)
df = data.frame(df_metrics$rate)
df <- plyr::rename(df, c("df_metrics.rate" = "rate"))

for (layer in layers){
  results = get_3comp_pca(bdd, weight, metric, layer, regularization)
  df = cbind(df,results)
}

colnames(df) <- c("rate",c(seq(1,60,by=1)))

ctrl = trainControl(method = "repeatedcv", number = 10) #10-fold cv
lambdas = 10^seq(2,-4,by=-0.1)
model1 = train( rate ~ ., data = df ,method = "glmnet", tuneGrid = expand.grid(alpha = 0, lambda = lambdas),preProc = c("center", "scale"),trControl = ctrl, metric = "Rsquared")

r_squared = model1$results$Rsquared[1]
r_squared








