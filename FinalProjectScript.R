# Load libraries
library(caret)
library(dplyr)
library(e1071)
library(randomForest)
library(nnet)
library(gbm3)
library(rpart)

# Load the data
df <- read.csv("shopping_trends_data2.csv")

#Remove id variable
df_noID <- select(df, -X, -Customer.ID) #dplyr

# Convert target to factor
df$frequency <- as.factor(df$frequency)

# Convert categorical variables to factors
factor_cols <- c("Gender", "item_purchased", "Category", "Location", "Size", "Color",
                 "Season", "subscription_status", "shipping_type", "discount_applied",
                 "promocode", "method")

df[factor_cols] <- lapply(df[factor_cols], as.factor)

# Normalize numeric columns
numeric_cols <- c("Age", "purchase_amount", "review_rating", "prev_purchases")
df[numeric_cols] <- scale(df[numeric_cols])

# Set seed for reproducibility
set.seed(1)

# Split data into training and test sets
train_index <- createDataPartition(df$frequency, p = 0.8, list = FALSE)
train_data <- df[train_index, ]
test_data <- df[-train_index, ]

# Set up cross-validation
control <- trainControl(method = "cv", number = 10)

# Model 1: Multinomial Logistic Regression
log_model <- train(frequency ~ ., data = train_data, method = "multinom", trControl = control)

# Model 2: Classification Tree
tree_model <- train(frequency ~ ., data = train_data, method = "rpart", trControl = control)

# Model 3: k-Nearest Neighbors
knn_model <- train(frequency ~ ., data = train_data, method = "knn", tuneLength = 10, trControl = control)

# Model 4: Naive Bayes
nb_model <- train(frequency ~ ., data = train_data, method = "naive_bayes", trControl = control)

# Model 5: Random Forest
rf_model <- train(frequency ~ ., data = train_data, method = "rf", trControl = control)

# Model 6: Gradient Boosting
gbm_model <- train(frequency ~ ., data = train_data, method = "gbm", verbose = FALSE, trControl = control)

# Compare models
results <- resamples(list(Logistic = log_model,
                          Tree = tree_model,
                          kNN = knn_model,
                          NaiveBayes = nb_model,
                          RandomForest = rf_model,
                          GBM = gbm_model))

summary(results)
dotplot(results)

# Evaluate best model on test data
best_model <- rf_model  # Example: you can switch this based on summary()
preds <- predict(best_model, newdata = test_data)
confusionMatrix(preds, test_data$frequency)