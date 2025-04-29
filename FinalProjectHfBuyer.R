# Predict_Weekly_Buyers.R
# Classify whether a customer is a high-frequency buyer (Yes/No)

# ------------------ Libraries ------------------

library(ggplot2)
library(dplyr) #Functions for editing data frames.
library(mosaic) # #Functions for common statistical tasks.
library(rpart) #Functions for creating trees.
library(rpart.plot) #Functions for plotting trees from rpart.
library(fastDummies) #Adds a function for making dummy variables.
library(caret) #Wide range of functions for machine learning algorithms.
library(pROC) #Performs ROC and AUC calculations.
library(gains) #Allows for Cumulative Lift Charts. 
library(FNN) #Functions for k Nearest Neighbor algorithms.
library(nnet) #Algorithms for creating neural network models.

# ------------------ Load and Prepare Data ------------------
Sdat <- read.csv("shopping_trends_data2.csv")

# Drop ID columns
Sdat_noID <- select(Sdat, -X, -Customer.ID)

# Create Binary Target 
Sdat_noID$hf_buyer <- ifelse(Sdat_noID$frequency %in% c('Weekly', 'Bi-Weekly'), 1, 0)


#Remove frequency
Sdat_noID <- select(Sdat_noID, -frequency)

#Remove extraneous categorical variables
Sdat_noID <- select(Sdat_noID, -item_purchased, -Category, -Season, -Location, -Size, -Color)

# Transform Yes/No to Binary
Sdat_noID$subscription_status <- ifelse(Sdat_noID$subscription_status == 'Yes', 1, 0)
Sdat_noID$discount_applied <- ifelse(Sdat_noID$discount_applied == 'Yes', 1, 0)
Sdat_noID$promocode <- ifelse(Sdat_noID$promocode == 'Yes', 1, 0)

#Make the binary categorical variables factor variables.
Sdat_noID_factor <- mutate(Sdat_noID,hf_buyer = as.factor(hf_buyer), subscription_status = as.factor(subscription_status),
                           discount_applied = as.factor(discount_applied), promocode = as.factor(promocode)) #dplyr

#Make dummy variables for multinomial variables.
#Remove the first dummy to avoid having all included.
Sdat_dum1 <- dummy_cols(Sdat_noID_factor, 
                        select_columns = c('Gender', 'shipping_type', 'method'),  
                        remove_first_dummy = TRUE) #fastDummies
#Remove original multinomial variables.
Sdat_dum2 <- select(Sdat_dum1, -Gender, -shipping_type, -method) #dplyr

#Normalize the scale variables.
#This prevents giving large-number variables too much influence.
Sdat_normdum <-mutate(Sdat_dum2, prev_purchases = (prev_purchases-min(prev_purchases))/(max(prev_purchases)-min(prev_purchases)))

# ------------------ Train/Test Split ------------------
#Create a partition using random sampling for training (70%) and test (30%)
#Include one value for each row of data in the data frame.
partition <- sample(c('train', 'test'), size=nrow(Sdat_noID),
                    replace=TRUE,prob=c(0.7,0.3)) #base R
#Add the partitioning variable to the dataframe as a new coluumn.
Sdat_newpart <- mutate(Sdat_normdum, partition) #dplyr


#Make training and test data frames based on value of 'partition'.
#Also, remove the 'partition' variable.
Sdat_train <- filter(Sdat_newpart, partition == 'train') %>% 
  select(-partition) #dplyr
Sdat_test <- filter(Sdat_newpart, partition == 'test') %>% 
  select(-partition) #dplyr


# ------------------ Logistic Regression ------------------
### BUILD LOGISTIC REGRESSION MODEL WITH STEPWISE PROCEDURE
#Make a model with no independent variables.
#This will be used to specify a starting point for the stepwise procedure.
ml_lognull <- glm(hf_buyer ~ 1, data=Sdat_train, family=binomial) #base R

#Make a model with all independent variables
ml_logall <- glm(hf_buyer ~., data=Sdat_train, family=binomial) #base R

#Create a model using the stepwise procedure.
#Use the model with all the ind. vars. to specify the maximum endpoint.
ml_logstep <- step(ml_lognull, scope=formula(ml_logall)) #base R
#Display the main results.  
summary(ml_logstep) #base R

#Evaluate Fit:  How good is the model on the training data? 
#Add unrounded predicted value to a new data frame in column 'Prediction'.
d_pred_logstep_train <- mutate(Sdat_train, Prediction = 
                                 predict(ml_logstep, Sdat_train, type="response")) #dplyr
#Add rounded predicted value to a new dataframe in column 'Prediction_Round'.
d_pred_logstep_train <- mutate(d_pred_logstep_train, Prediction_Round = 
                                 round(Prediction)) #dplyr

# Convert predictions to factor with both levels, even if 1 is missing
d_pred_logstep_train$Prediction_Round <- factor(d_pred_logstep_train$Prediction_Round,
                                                levels = c(0, 1))

#Compute/display the Percentage Correct in the training data to evaluate fit.
mean(~(hf_buyer == Prediction_Round), data=d_pred_logstep_train) #mosaic

#Create/display Classification Tables for the training data.
#Classification Table of raw counts.
tally(hf_buyer ~ Prediction_Round, data=d_pred_logstep_train) %>% addmargins() #mosaic
#Classification Table of percentages of training data.
tally(hf_buyer ~ Prediction_Round, data=d_pred_logstep_train) %>% 
  prop.table(margin=1) %>% round(2) #mosaic 


### TEST THE STEPWISE LOGISTIC MODEL WITH TEST DATA
#Evaluate Accuracy:  How good is the model on the test data? 
#Add unrounded predicted value to a new dataframe in column 'Prediction'.
d_pred_logstep_test <- mutate(Sdat_test, Prediction = 
                                predict(ml_logstep, Sdat_test, type="response")) #dplyr
#Add rounded predicted value to a new dataframe in column 'Prediction_Round'.
d_pred_logstep_test <- mutate(d_pred_logstep_test, Prediction_Round = 
                                round(Prediction)) #dplyr

#Compute/display the Percentage Correct in test data to evaluate accuracy.
mean(~(hf_buyer == Prediction_Round), data=d_pred_logstep_test) #mosaic

#Create/display Classification Tables for the test data.
#Classification Table of raw counts.
tally(hf_buyer ~ Prediction_Round, data=d_pred_logstep_test) %>% addmargins() #mosaic
#Classification Table of percentages of test data.
tally(hf_buyer ~ Prediction_Round, data=d_pred_logstep_test) %>% 
  prop.table(margin=1) %>% round(2) #mosaic 


# ------------------ Classification Tree ------------------
### BUILD TREE MODEL USING K-FOLD CROSS VALIDATION

#Set the Seed.
#Set the seed before creating the validation to ensure consistent results.
set.seed(1) 

#Run the k-fold cross validation.
#Run a tree using 10-fold cross validation.
ml_kfold_tree <- train(hf_buyer~., data= Sdat_train, method="rpart",
                       trControl=trainControl("cv",number=10), tuneLength=8) #caret


model.matrix(hf_buyer ~ ., data = Sdat_train)
Sdat_train$hf_buyer <- droplevels(Sdat_train$hf_buyer)

#Display information about optimized complexity parameter.
ml_kfold_tree #base R
#Show more info about CP and accuracy.
plot(ml_kfold_tree) #base R

### BUILD CLASSIFICATION TREE WITH A COMPLEXITY PARAMETER OF 0.1428571 

#Build a model with CP = 0.1428571 
ml_treekfold <- rpart(hf_buyer ~., data=Sdat_train, 
                      method="class", cp=0.1428571)

#Plot the tree.
rpart.plot(ml_treekfold,roundint=FALSE,nn=TRUE,extra=1) #rpart.plot
#Plot again, but extra=4 gives percentages of each outcome in each node.
rpart.plot(ml_treekfold,roundint=FALSE,nn=TRUE,extra=4) #rpart.plot

#Evaluate Accuracy:  How good is the model on the test data? 
#Get predictions for the tree model in the test data
preds_treekfold_test <- predict(ml_treekfold, Sdat_test, type="class")
#Calculate percentage correct for the tree01 model in the test data
mean(~(hf_buyer == preds_treekfold_test), data = Sdat_test)

#Create/display Classification Tables for test data.
#Classification Table of raw counts.
tally(hf_buyer ~ preds_treekfold_test, data = Sdat_test) %>% addmargins()
#Classification Table of percentages of test data.
tally(hf_buyer ~ preds_treekfold_test, data = Sdat_test) %>% prop.table(margin =1) %>%round(2)