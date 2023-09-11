#####################################################################################
# 1. DESCRIPTION:
#####################################################################################
#Code to obtain the results shown in Figure 1
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
library("ggplot2")
setwd("") #write here the working directory
#####################################################################################
# 3. PARAMETERS:
#####################################################################################
model_name <- 'VGG16' #don't change this parameter
bdd <- c('CFD') #change this parameter with the database of interest
weight <- c('imagenet') #don't change this parameter
metric <- c('gini_flatten')  #don't change this parameter
#####################################################################################
# 4. DATA MANAGEMENT
#####################################################################################
#These lines of code are used to perform data management tasks to ensure that data is in the format required for statistical processing functions.

labels_path =paste('../../data/redesigned/',bdd,'/labels_',bdd,'.csv',sep='') #use the path of the data obtain with scripts contain in *code/pre_trained_models*, that will have enabled you to obtain the Gini index (or PCA data) on the data in your databases
log_path = paste('../../results/',bdd,'/log_',sep="")

matrix_metrics <- do.call(cbind, fromJSON(file = paste(log_path,'_',bdd,'_',weight,'_',metric,'_','_BRUTMETRICS','.csv',sep=""),simplify = FALSE))
      
colnames(matrix_metrics)[2] <- 'input_1'
        
df_metrics <- as.data.frame(matrix_metrics, optional = TRUE)
     
df_metrics = sapply(df_metrics, as.numeric)

df_metrics <- as.data.frame(df_metrics)

df = plyr::rename(df_metrics, c("input_1" = "input_1",
                                         'block1_conv1'='conv1_1','block1_conv2'='conv1_2','block1_pool'='pool1',
                                         'block2_conv1'='conv2_1','block2_conv2'='conv2_2','block2_pool'='pool2',
                                         'block3_conv1'='conv3_1','block3_conv2'='conv3_2','block3_conv3'='conv3_3','block3_pool'='pool3',
                                         'block4_conv1'='conv4_1','block4_conv2'='conv4_2','block4_conv3'='conv4_3','block4_pool'='pool4',
                                         'block5_conv1'='conv5_1','block5_conv2'='conv5_2','block5_conv3'='conv5_3','block5_pool'='pool5',
                                         'flatten'='flatten','fc1'='fc6_relu','fc2'='fc7_relu'))

scaled_df <- scale(df[,-1]) 
df <- cbind(df$rate ,scaled_df)
df<- as.data.frame(df, optional = TRUE)
df <- plyr::rename(df, c("V1" = "rate"))

#####################################################################################
# 5. MODEL: 10-fold CROSS VALIDATION
#####################################################################################

#These lines of code are used to perform 10-fold cross validation, repeated 100 times, on the Gini indices of the layers of interest. 

layers = c('conv1_1','conv1_2',
           'conv2_1','conv2_2',
           'conv3_1','conv3_2','conv3_3',
           'conv4_1','conv4_2','conv4_3',
           'conv5_1','conv5_2','conv5_3',
           'fc6_relu','fc7_relu')

ctrl = trainControl(method = "repeatedcv", number = 10, repeats = 100) #10-fold cv with 100 repetitions


#conv1_1
model_conv1_1 <- train(rate ~ 
                 conv1_1  ,
               data = df, method = "lm", trControl = ctrl)
conv1_1 <- c(model_conv1_1$results$Rsquared)
conv1_1_cor <- unname(model_conv1_1$finalModel$coefficients[2]) #récupérer l'effect size
print(summary(model_conv1_1))
#conv1_2
model_conv1_2 <- train(rate ~ 
                 conv1_2   ,
               data = df, method = "lm", trControl = ctrl)
conv1_2 <- c(model_conv1_2$results$Rsquared)
conv1_2_cor <- unname(model_conv1_2$finalModel$coefficients[2])
print(summary(model_conv1_2))
#conv2_1
model_conv2_1 <- train(rate ~ 
                 conv2_1  ,
               data = df, method = "lm", trControl = ctrl)
conv2_1 <- c(model_conv2_1$results$Rsquared)
conv2_1_cor <- unname(model_conv2_1$finalModel$coefficients[2])
print(summary(model_conv2_1))
#conv2_2
model_conv2_2 <- train(rate ~ 
                 conv2_2  ,
               data = df, method = "lm", trControl = ctrl)
conv2_2 <- c(model_conv2_2$results$Rsquared)
conv2_2_cor <- unname(model_conv2_2$finalModel$coefficients[2])
print(summary(model_conv2_2))
#conv3_1
model_conv3_1 <- train(rate ~ 
                 conv3_1  ,
               data = df, method = "lm", trControl = ctrl)
conv3_1 <-c(model_conv3_1$results$Rsquared)
conv3_1_cor <- unname(model_conv3_1$finalModel$coefficients[2])
print(summary(model_conv3_1))
#conv3_2
model_conv3_2 <- train(rate ~ 
                 conv3_2 ,
               data = df, method = "lm", trControl = ctrl)
conv3_2 <-c(model_conv3_2$results$Rsquared)
conv3_2_cor <- unname(model_conv3_2$finalModel$coefficients[2])
print(summary(model_conv3_2))
#conv3_3
model_conv3_3 <- train(rate ~ 
                 conv3_3  ,
               data = df, method = "lm", trControl = ctrl)
conv3_3 <-c(model_conv3_3$results$Rsquared)
conv3_3_cor <- unname(model_conv3_3$finalModel$coefficients[2])
print(summary(model_conv3_3))
#conv4_1
model_conv4_1 <- train(rate ~ 
                 conv4_1 ,
               data = df, method = "lm", trControl = ctrl)
conv4_1 <-c(model_conv4_1$results$Rsquared)
conv4_1_cor <- unname(model_conv4_1$finalModel$coefficients[2])
print(summary(model_conv4_1))
#conv4_2
model_conv4_2 <- train(rate ~ 
                 conv4_2   ,
               data = df, method = "lm", trControl = ctrl)
conv4_2 <-c(model_conv4_2$results$Rsquared)
conv4_2_cor <- unname(model_conv4_2$finalModel$coefficients[2])
print(summary(model_conv4_2))
#conv4_3
model_conv4_3 <- train(rate ~ 
                 conv4_3  ,
               data = df, method = "lm", trControl = ctrl)
conv4_3 <-c(model_conv4_3$results$Rsquared)
conv4_3_cor <-unname(model_conv4_3$finalModel$coefficients[2])
print(summary(model_conv4_3))
#conv5_1
model_conv5_1 <- train(rate ~ 
                 conv5_1  ,
               data = df, method = "lm", trControl = ctrl)
conv5_1 <-c(model_conv5_1$results$Rsquared)
conv5_1_cor <- unname(model_conv5_1$finalModel$coefficients[2])
print(summary(model_conv5_1))
#conv5_2
model_conv5_2 <- train(rate ~ 
                 conv5_2 ,
               data = df, method = "lm", trControl = ctrl)
conv5_2 <-c(model_conv5_2$results$Rsquared)
conv5_2_cor <- unname(model_conv5_2$finalModel$coefficients[2])
print(summary(model_conv5_2))
#conv5_3
model_conv5_3 <- train(rate ~ 
                 conv5_3  ,
               data = df, method = "lm", trControl = ctrl)
conv5_3 <-c(model_conv5_3$results$Rsquared)
conv5_3_cor <- unname(model_conv5_3$finalModel$coefficients[2])
print(summary(model_conv5_3))
#fc6_relu
model_fc6_relu <- train(rate ~ 
                 fc6_relu  ,
               data = df, method = "lm", trControl = ctrl)
fc6_relu <- c(model_fc6_relu$results$Rsquared)
fc6_relu_cor <-unname(model_fc6_relu$finalModel$coefficients[2])
print(summary(model_fc6_relu))
#fc7_relu
model_fc7_relu <- train(rate ~ 
                 fc7_relu,
               data = df, method = "lm", trControl = ctrl)
fc7_relu <- c(model_fc7_relu$results$Rsquared)
fc7_relu_cor <- unname(model_fc7_relu$finalModel$coefficients[2])
print(summary(model_fc7_relu))
#graphe de l'évolution des R2 de chaque modèle
r_sq = data.frame(conv1_1,conv1_2,
                  conv2_1,conv2_2,
                  conv3_1,conv3_2,conv3_3,
                  conv4_1,conv4_2,conv4_3,
                  conv5_1,conv5_2,conv5_3,
                  fc6_relu,fc7_relu)
cors = c(conv1_1_cor,conv1_2_cor,
                  conv2_1_cor,conv2_2_cor,
                  conv3_1_cor,conv3_2_cor,conv3_3_cor,
                  conv4_1_cor,conv4_2_cor,conv4_3_cor,
                  conv5_1_cor,conv5_2_cor,conv5_3_cor,
                  fc6_relu_cor,fc7_relu_cor)


r_sq = t(r_sq)
r_sq = data.frame((r_sq))

r_sq$cors = cors
r_sq$names = layers
rownames(r_sq) <- r_sq$names
r_sq$names <- NULL

colnames(r_sq) = c("rsq","effect_size")

#Here, the results are recorded and plotted on a graph.

write.csv(r_sq,paste("plot_",bdd,".csv"),row.names = TRUE)

barplot(t(r_sq), las = 2, main = paste('R2 for ',bdd,", sparseness only, 10-fold CV",sep=""),ylim=c(0,0.20))
barplot(t(cors), las = 2, main = paste('Correlations for ',bdd,", sparseness only, 10-fold CV",sep=""),ylim=c(-0.4,0.4))





