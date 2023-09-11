#####################################################################################
# 1. DESCRIPTION:
#####################################################################################
#Code to obtain the results shown in tab.1 (Gini columns)
#####################################################################################
# 2. LIBRAIRIES:
#####################################################################################
library("rjson")
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
setwd("") #working directory
#####################################################################################
# 3. PARAMETERS:
#####################################################################################
model_name <- 'VGG16' #don't change this parameter
bdd <- c('SCUT-FBP') #change this parameter with the database of interest
weight <- c('imagenet')#don't change this parameter
metric <- c('gini_flatten')#don't change this parameter
regularization = 'ridge'#don't change this parameter
#####################################################################################
# 4. DATA MANAGEMENT
#####################################################################################
#These lines of code are used to perform data management tasks to ensure that data is in the format required for statistical processing functions.
labels_path = paste('../../data/redesigned/',bdd,'/labels_',bdd,'.csv', sep="")
log_path =paste('../../results/',bdd,'/log_', sep="")

matrix_metrics <- do.call(cbind, fromJSON(file = paste(log_path,'_',bdd,'_',weight,'_',metric,'_','_BRUTMETRICS','.csv',sep=""),simplify = FALSE))

colnames(matrix_metrics)[2] <- 'input_1'

matrix_complexity <- do.call(cbind, fromJSON(file = paste(log_path,'_',bdd,'_',weight,'_','mean','_','_BRUTMETRICS','.csv',sep=""),simplify = FALSE))
colnames(matrix_complexity)[2] <- 'input_1'

df_metrics <- as.data.frame(matrix_metrics, optional = TRUE)
df_complexity <- as.data.frame(matrix_complexity, optional = TRUE)        
df_metrics = sapply(df_metrics, as.numeric)
df_complexity = sapply(df_complexity, as.numeric)

df_metrics <- as.data.frame(df_metrics)
df_complexity <- as.data.frame(df_complexity[,-1])

df_metrics = plyr::rename(df_metrics, c("input_1" = "input_1",
                                        'block1_conv1'='conv1_1','block1_conv2'='conv1_2','block1_pool'='pool1',
                                        'block2_conv1'='conv2_1','block2_conv2'='conv2_2','block2_pool'='pool2',
                                        'block3_conv1'='conv3_1','block3_conv2'='conv3_2','block3_conv3'='conv3_3','block3_pool'='pool3',
                                        'block4_conv1'='conv4_1','block4_conv2'='conv4_2','block4_conv3'='conv4_3','block4_pool'='pool4',
                                        'block5_conv1'='conv5_1','block5_conv2'='conv5_2','block5_conv3'='conv5_3','block5_pool'='pool5',
                                        'flatten'='flatten','fc1'='fc6_relu','fc2'='fc7_relu'))
df_complexity = plyr::rename(df_complexity, c("input_1" = "input_1_comp",
                                              'block1_conv1'='conv1_1_comp','block1_conv2'='conv1_2_comp','block1_pool'='pool1_comp',
                                              'block2_conv1'='conv2_1_comp','block2_conv2'='conv2_2_comp','block2_pool'='pool2_comp',
                                              'block3_conv1'='conv3_1_comp','block3_conv2'='conv3_2_comp','block3_conv3'='conv3_3_comp','block3_pool'='pool3_comp',
                                              'block4_conv1'='conv4_1_comp','block4_conv2'='conv4_2_comp','block4_conv3'='conv4_3_comp','block4_pool'='pool4_comp',
                                              'block5_conv1'='conv5_1_comp','block5_conv2'='conv5_2_comp','block5_conv3'='conv5_3_comp','block5_pool'='pool5_comp',
                                              'flatten'='flatten_comp','fc1'='fc6_relu_comp','fc2'='fc7_relu_comp'))


df <- cbind(df_metrics, df_complexity)

scaled_df <- scale(df[,-1]) 
df <- cbind(df$rate ,scaled_df) 
df<- as.data.frame(df, optional = TRUE)
df <- plyr::rename(df, c("V1" = "rate"))


df = df[,c(1,3,4,6,7,9,10,11,13,14,15,17,18,19,22,23)] #chose column number of interest (Gini only, or Gini + PCA, or PCA only )

#####################################################################################
# 5. MODEL: RIDGE REGRESSION
#####################################################################################

#10-fold cv with 100 repetitions
ctrl = trainControl(method = "repeatedcv", number = 10, repeats = 100) 

lambdas = 10^seq(2,-4,by=-0.1)

model = train( rate ~ ., data = df ,method = "glmnet", tuneGrid = expand.grid(alpha = 0, lambda = lambdas),preProc = c("center", "scale"),trControl = ctrl, metric = "Rsquared") #alpha = 1 pour ridge (0 pour lasso)
r_squared = model$results$Rsquared[1]

print(r_squared)


#####################################################################################
# 6. Effect size
#####################################################################################

coeffs = coef(model$finalModel, model$bestTune$lambda)

#for sparsity form Gini index only (change column names for values of other columsn(PCA, or PCA + Gini)
spars1_1_r = coeffs@x[2]
spars1_2_r = coeffs@x[3]
spars2_1_r = coeffs@x[4]
spars2_2_r = coeffs@x[5]
spars3_1_r = coeffs@x[6]
spars3_2_r = coeffs@x[7]
spars3_3_r = coeffs@x[8]
spars4_1_r = coeffs@x[9]
spars4_2_r = coeffs@x[11]
spars4_3_r = coeffs@x[11]
spars5_1_r = coeffs@x[12]
spars5_2_r = coeffs@x[13]
spars5_3_r = coeffs@x[14]
sparsfc6_r = coeffs@x[15]
sparsfc7_r = coeffs@x[16]

xlab = c( 'spars1_1', 'spars1_2',
          'spars2_1', 'spars2_2',
          'spars3_1', 'spars3_2','spars3_3',
          'spars4_1', 'spars4_2','spars4_3',
          'spars5_1', 'spars5_2','spars5_3',
          'sparsfc6', 'sparsfc7')

effect_size_order = c( spars1_1_r, spars1_2_r, 
                       spars2_1_r, spars2_2_r,
                       spars3_1_r, spars3_2_r, spars3_3_r,
                       spars4_1_r, spars4_2_r, spars4_3_r,
                       spars5_1_r, spars5_2_r, spars5_3_r,
                       sparsfc6_r, sparsfc7_r)

barplot(effect_size_order,  names = xlab, las = 2, col = "yellow" , main = paste('Effect size for ',bdd,' with ',regularization," regularization" ,", sparseness only",sep=""))

mean(effect_size_order)

