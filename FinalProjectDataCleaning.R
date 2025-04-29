# Full_EDA_Shopping_Trends.R
# Comprehensive EDA for Shopping Trends Purchase Frequency Prediction

# ------------------ Libraries ------------------
library(dplyr)
library(ggplot2)
library(reshape2)

# ------------------ Load and Clean Data ------------------
df <- read.csv("shopping_trends_data2.csv")

#Remove Specific Identifiers
df_noID <- select(df, -X, -Customer.ID)

#Make the dependent variable a factor variable.
df_factor <- mutate(df_noID,frequency = as.factor(frequency)) #dplyr

# Define column types
factor_cols <- c('Gender', 'item_purchased', 'Category', 'Location', 'Size', 'Color',
                 'Season', 'subscription_status', 'shipping_type', 'discount_applied',
                 'promocode', 'method', 'frequency')
numeric_cols <- c('Age', 'purchase_amount', 'review_rating', 'prev_purchases')

# Convert column types
df[factor_cols] <- lapply(df[factor_cols], as.factor)

# ------------------ 1. Summary Statistics ------------------
cat("===== SUMMARY STATISTICS (Numeric Features) =====\n")
print(summary(df[numeric_cols]))

cat("\n===== FREQUENCY DISTRIBUTION (Categorical Features) =====\n")
for (col in factor_cols) {
  cat("\n---", col, "---\n")
  print(summary(df[[col]]))
}

# ------------------ 2. Missing Value Check ------------------
cat("\n===== MISSING VALUES =====\n")
missing_vals <- sapply(df, function(x) sum(is.na(x)))
print(missing_vals)

# ------------------ 3. Outlier Detection (IQR Method) ------------------
cat("\n===== OUTLIER COUNTS (IQR Method) =====\n")
detect_outliers <- function(vec) {
  q1 <- quantile(vec, 0.25, na.rm = TRUE)
  q3 <- quantile(vec, 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  sum(vec < (q1 - 1.5 * iqr) | vec > (q3 + 1.5 * iqr))
}

for (col in numeric_cols) {
  out_count <- detect_outliers(df[[col]])
  cat(sprintf("%s: %d outliers\n", col, out_count))
}

# ------------------ 4. Target Distribution ------------------
ggplot(df, aes(x = frequency)) +
  geom_bar(fill = "skyblue") +
  theme_minimal() +
  ggtitle("Distribution of Purchase Frequency") +
  xlab("Frequency") +
  ylab("Count")

# ------------------ 5. Numeric Features by Frequency ------------------
for (col in numeric_cols) {
  print(
    ggplot(df, aes(x = frequency, y = .data[[col]])) +
      geom_boxplot(fill = "lightblue") +
      theme_minimal() +
      ggtitle(paste("Boxplot of", col, "by Purchase Frequency")) +
      xlab("Frequency") +
      ylab(col)
  )
}

# ------------------ 6. Outlier Visualization ------------------
for (col in numeric_cols) {
  p <- ggplot(df, aes_string(y = col)) +
    geom_boxplot(fill = "salmon") +
    theme_minimal() +
    ggtitle(paste("Boxplot for Outlier Detection:", col))
  print(p)
}

# ------------------ 7. Categorical Feature Breakdown ------------------
cat_features <- c("Gender", "Season", "Category", "shipping_type")
for (col in cat_features) {
  print(
    ggplot(df, aes_string(x = col, fill = "frequency")) +
      geom_bar(position = "fill") +
      scale_y_continuous(labels = scales::percent) +
      theme_minimal() +
      ggtitle(paste("Proportional Frequency by", col)) +
      ylab("Proportion")
  )
}

# ------------------ 8. Correlation Heatmap ------------------
cor_matrix <- cor(df[, numeric_cols], use = "complete.obs")
melted_corr <- melt(cor_matrix)

print(
  ggplot(melted_corr, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  geom_text(aes(label = round(value, 2)), color = "black") +
  scale_fill_gradient2(low = "red", high = "blue", mid = "white",
                       midpoint = 0, limit = c(-1,1), name="Correlation") +
  theme_minimal() +
  ggtitle("Correlation Heatmap (Manual)"))

# ------------------ 9. PCA Visualization ------------------
# 1. Select numeric columns and scale them
df_scaled <- scale(df[, numeric_cols])

# 2. Perform PCA
pca <- prcomp(df_scaled, center = TRUE, scale. = TRUE)

# 3. Create a data frame with the first 2 principal components
pca_df <- data.frame(PC1 = pca$x[, 1],
                     PC2 = pca$x[, 2],
                     Frequency = df$frequency)

# 4. Plot using ggplot2
ggplot(pca_df, aes(x = PC1, y = PC2, color = Frequency)) +
  geom_point(alpha = 0.7, size = 2) +
  theme_minimal() +
  ggtitle("PCA: First Two Principal Components") +
  xlab(paste0("PC1 (", round(summary(pca)$importance[2,1]*100, 1), "%)")) +
  ylab(paste0("PC2 (", round(summary(pca)$importance[2,2]*100, 1), "%)")) +
  scale_color_brewer(palette = "Dark2")

# ------------------ 10. Summary Statistics Factor Features ------------------
cat("===== SUMMARY STATISTICS (Factor Features) =====\n")
print(summary(df[factor_cols]))
