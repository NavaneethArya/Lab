# Combining and Merging datasets 
df1 <- data.frame(ID = 1:5, Name = c("A", "B", "C", "D", "E")) 
df2 <- data.frame(ID = 3:7, Age = c(25, 28, 22, 30, 26)) 
# Merging on ID 
merged_df<- merge(df1, df2, by = "ID", all = TRUE) 
print(merged_df) 
# Reshaping data (pivoting) 
library(tidyr) 
long_df<- gather(merged_df, key = "Variable", value = "Value", -ID) 
print(long_df) 
# Pivoting back to wide format 
wide_df<- spread(long_df, key = "Variable", value = "Value") 
print(wide_df)

# String Manipulation 
strings<- c("Data_Analysis", "Computational_Statistics") 
library(stringr) 
# Replace underscores with spaces 
strings<- str_replace_all(strings, "_", " ") 
print(strings) 
# Regular Expressions 
emails<- c("user1@example.com", "user2@gmail.com") 
valid_emails<- grep("@example.com", emails, value = TRUE) 
print(valid_emails)

# Load required libraries  
library(dplyr)  
library(lubridate)  
library(forecast)  
# Sample data creation  
set.seed(123)  
date_seq <- seq(from = as.Date("2020-01-01"), by = "month", length.out = 36)  
data <- data.frame(  
Date = date_seq,  
Value1 = rnorm(36, mean = 200, sd = 20),  
Value2 = rnorm(36, mean = 100, sd = 10)  
)  
# Display the original data  
print("Original Data:")  
print(data)  
# Convert to time series  
data_ts <- ts(data[,-1], frequency = 12, start = c(2020, 1))  
# Grouping by year and calculating mean for each variable  
grouped_data <- data %>%  
mutate(Year = year(Date)) %>%  
group_by(Year) %>%  
summarise(Mean_Value1 = mean(Value1), 
Mean_Value2 = mean(Value2))  
# Display grouped data  
print("Grouped Data:")  
print(grouped_data)  
# Multivariate time series handling  
# This creates a time series object for both Value1 and Value2  
multivariate_ts <- ts(data[, -1], start = c(2020, 1), frequency = 12)  
# Forecasting with ARIMA for Value1  
fit_value1 <- auto.arima(multivariate_ts[, 1])  
forecast_value1 <- forecast(fit_value1, h = 6) # Forecasting for next 6 months  
# Forecasting with ARIMA for Value2  
fit_value2 <- auto.arima(multivariate_ts[, 2])  
forecast_value2 <- forecast(fit_value2, h =6) # Forecasting for next 6 months  
# Plot the forecasts  
par(mfrow = c(2, 1)) # Arrange plots vertically  
plot(forecast_value1, main = "Forecast for Value1", xlab = "Time", ylab = "Value1")  
plot(forecast_value2, main = "Forecast for Value2", xlab = "Time", ylab = "Value2")  
# Reset plot layout  
par(mfrow = c(1,2))  
print(forecast_value1) 
print(forecast_value2)  
checkresiduals(fit_value1) 


data<- c(10, 20, 30, 40, 50, 60, 70, 80, 90, 100) 
# Central Tendency 
mean_value<- mean(data) 
median_value<- median(data) 
mode_value<- as.numeric(names(sort(table(data), decreasing=TRUE)[1])) 
# Dispersion Measures 
sd_value<- sd(data) 
var_value<- var(data) 
mad_value<- mad(data) 
quartile_deviation<- IQR(data) / 2 
list(mean = mean_value, median = median_value, mode = mode_value,  
sd = sd_value, variance = var_value, MAD = mad_value, quartile_dev = 
quartile_deviation)



# Load necessary libraries 
library(caret)     
# For cross-validation functions 
library(Metrics)   # For performance metrics 
library(dplyr)     
# For data manipulation 
# Load dataset 
data(mtcars) 
# Set the target variable and predictor variables 
target_variable<- "mpg" 
predictors<- setdiff(names(mtcars), target_variable) 
# Split the data into training and validation sets (80% training, 20% validation) 
set.seed(123)  # For reproducibility 
trainIndex<- createDataPartition(mtcars$mpg, p = 0.8, list = FALSE) 
trainData<- mtcars[trainIndex, ] 
validData<- mtcars[-trainIndex, ] 
# Function to calculate RMSE, MAE, and R2 
calc_metrics<- function(actual, predicted) { 
rmse<- rmse(actual, predicted) 
mae<- mae(actual, predicted) 
r2 <- cor(actual, predicted)^2  # R-squared 
return(c(RMSE = rmse, MAE = mae, R2 = r2)) 
} 
# Validation Set Approach 
model_val<- lm(mpg ~ ., data = trainData) 
pred_val<- predict(model_val, newdata = validData) 
metrics_val<- calc_metrics(validData$mpg, pred_val) 
print("Validation Set Metrics:") 
print(metrics_val) 
# Leave-One-Out Cross-Validation (LOOCV) 
loocv_metrics<- sapply(1:nrow(mtcars), function(i) { 
train_loocv<- mtcars[-i, ] 
test_loocv<- mtcars[i, , drop = FALSE] 
model_loocv<- lm(mpg ~ ., data = train_loocv) 
pred_loocv<- predict(model_loocv, newdata = test_loocv) 
calc_metrics(test_loocv$mpg, pred_loocv) 
}) 
loocv_metrics_avg<- colMeans(loocv_metrics) 
print("LOOCV Metrics (Average across folds):") 
print(loocv_metrics_avg) 
# K-Fold Cross-Validation (5-fold) 
k <- 5 
cv_results<- train(mpg ~ ., data = mtcars, method = "lm", 
trControl = trainControl(method = "cv", number = k, 
summaryFunction = defaultSummary)) 
print("K-Fold Cross-Validation Metrics:") 
print(cv_results$results) 



# Load required libraries 
if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2") 
if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr") 
library(ggplot2) 
library(dplyr) 
# Set seed for reproducibility 
set.seed(123) 
# Create a frequency distribution as a data frame 
# Example: Frequency distribution of a survey response 
response_data <- data.frame( 
response = c("Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"), 
frequency = c(5, 15, 25, 30, 25) 
) 
# Display the frequency distribution 
print("Frequency Distribution:") 
print(response_data) 
# Calculate proportions 
response_data <- response_data %>% 
mutate(proportion = frequency / sum(frequency)) 
# Normal Distribution parameters 
mu <- mean(1:length(response_data$response)) # Mean of the response categories 
sigma <- sd(1:length(response_data$response)) # Standard deviation 
# Binomial Distribution parameters 
n <- sum(response_data$frequency)  # Total number of responses 
p <- mean(response_data$proportion) # Probability of success 
# Poisson Distribution parameter (lambda) 
lambda <- n * p 
# Bernoulli Distribution probabilities 
p_bernoulli <- response_data$proportion[1]  # Probability of "Strongly Disagree" 
# Generate values for distributions 
x <- 0:(max(response_data$frequency) + 10) 
# Create data frame for Normal distribution 
normal_df <- data.frame( 
x = x, 
density = dnorm(x, mean = mu, sd = sigma) 
) 
# Create data frame for Binomial distribution 
binomial_df <- data.frame( 
x = x, 
density = dbinom(x, size = n, prob = p) 
) 
# Create data frame for Poisson distribution 
poisson_df <- data.frame( 
x = x, 
density = dpois(x, lambda = lambda) 
) 
# Create data frame for Bernoulli distribution 
bernoulli_df <- data.frame( 
x = c(0, 1), 
density = c(1 - p_bernoulli, p_bernoulli) # 0 for failure, 1 for success 
) 
# Visualize distributions 
ggplot() + 
geom_line(data = normal_df, aes(x = x, y = density), color = "blue") + 
geom_line(data = binomial_df, aes(x = x, y = density), color = "red") + 
geom_line(data = poisson_df, aes(x = x, y = density), color = "green") + 
geom_bar(data = bernoulli_df, aes(x = factor(x), y = density), stat = "identity", fill = 
"purple", alpha = 0.5) + 
labs(title = "Probability Distributions", 
x = "Response Categories / Counts", 
y = "Density / Probability") + 
scale_x_discrete(labels = c("Failure", "Success")) + 
theme_minimal() + 
theme(legend.position = "top") + 
scale_color_manual(values = c("blue", "red", "green")) 
# Display summary statistics 
cat("Summary Statistics for the Frequency Distribution:\n") 
cat("Mean of Frequencies: ", mean(response_data$frequency), "\n") 
cat("Standard Deviation of Frequencies: ", sd(response_data$frequency), "\n") 
cat("Total Responses: ", sum(response_data$frequency), "\n") 
cat("Probability of 'Strongly Disagree': ", p_bernoulli, "\n") 
cat("Mean for Normal Distribution (mu): ", mu, "\n") 
cat("Standard Deviation for Normal Distribution (sigma): ", sigma, "\n") 
cat("Lambda for Poisson Distribution: ", lambda, "\n")


# Set seed for reproducibility 
set.seed(123) 
# One-sample data 
one_sample_data<- rnorm(30, mean = 50, sd = 10) 
# Two-sample data 
two_sample_data1 <- rnorm(30, mean = 55, sd = 10) 
two_sample_data2 <- rnorm(30, mean = 50, sd = 10) 
# Paired sample data 
paired_data_before<- rnorm(30, mean = 60, sd = 10) 
paired_data_after<- paired_data_before + rnorm(30, mean = -2, sd = 5) 
# One-sample t-test 
one_sample_test<- t.test(one_sample_data, mu = 50) 
cat("One-Sample t-Test Results:\n") 
print(one_sample_test) 
# Two-sample t-test 
two_sample_test<- t.test(two_sample_data1, two_sample_data2) 
cat("\nTwo-Sample t-Test Results:\n") 
print(two_sample_test) 
# Paired t-test 
paired_test<- t.test(paired_data_before, paired_data_after, paired = TRUE) 
cat("\nPaired Sample t-Test Results:\n") 
print(paired_test) 


# Load necessary library 
library(dplyr) 
# One-Way ANOVA 
set.seed(123)  # For reproducibility 
treatment_A<- c(23, 25, 20, 22, 26) 
treatment_B<- c(30, 29, 31, 32, 28) 
treatment_C<- c(35, 36, 34, 33, 37) 
data_one_way<- data.frame( 
value = c(treatment_A, treatment_B, treatment_C), 
treatment = factor(rep(c("A", "B", "C"), each = 5)) 
) 
one_way_anova<- aov(value ~ treatment, data = data_one_way) 
print(summary(one_way_anova)) 
# Two-Way ANOVA 
treatment_A_male<- c(23, 25, 20, 22, 26) 
treatment_A_female<- c(30, 28, 31, 29, 32) 
treatment_B_male<- c(27, 29, 30, 26, 25) 
treatment_B_female<- c(35, 36, 34, 33, 37) 
data_two_way<- data.frame( 
value = c(treatment_A_male, treatment_A_female, treatment_B_male, 
treatment_B_female), 
treatment = factor(rep(c("A", "B"), each = 10)), 
gender = factor(rep(c("Male", "Female"), times = 10)) 
) 
two_way_anova<- aov(value ~ treatment * gender, data = data_two_way) 
print(summary(two_way_anova)) 



# Load necessary libraries 
library(ggplot2)   # For plotting 
library(dplyr)     
# For data manipulation 
library(reshape2)  # For reshaping data for heatmaps 
library(corrplot)  # For correlation plot 
# Load the mtcars dataset 
data(mtcars) 
# 1. Correlation 
correlation_matrix<- cor(mtcars) 
print("Correlation Matrix:") 
print(correlation_matrix) 
# 2. Rank Correlation (Spearman) 
rank_correlation_matrix<- cor(mtcars, method = "spearman") 
print("Rank Correlation Matrix (Spearman):") 
print(rank_correlation_matrix) 
# 3. Linear Regression Example 
# Let's predict mpg based on wt and hp 
model<- lm(mpg ~ wt + hp, data = mtcars) 
summary(model) 
# Predictions and residuals 
mtcars$predicted_mpg<- predict(model) 
mtcars$residuals<- residuals(model) 
# 4. Plotting x-y plot 
# Scatter plot of actual vs predicted mpg 
ggplot(mtcars, aes(x = predicted_mpg, y = mpg)) + 
geom_point(color = 'blue') + 
geom_smooth(method = 'lm', color = 'red') + 
labs(title = "Actual vs Predicted MPG", 
x = "Predicted MPG", 
y = "Actual MPG") + 
theme_minimal() 
# 5. Heatmap of Correlation Matrix 
# Reshape the correlation matrix 
cor_melted<- melt(correlation_matrix) 
# Create the heatmap 
ggplot(cor_melted, aes(Var1, Var2, fill = value)) + 
geom_tile() + 
scale_fill_gradient2(low = "blue", high = "red", mid = "white",  
midpoint = 0, limit = c(-1, 1), space = "Lab",  
name="Correlation") + 
theme_minimal() + 
labs(title = "Heatmap of Correlation Matrix") + 
theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) 
# 6. Correlation Plot 
corrplot(correlation_matrix, method = "circle", type = "upper",  
order = "hclust", tl.col = "black", tl.srt = 45,  
title = "Correlation Plot") 


install.packages(c("mlbench", "ggplot2", "dplyr", "factoextra")) 
library(mlbench) 
library(ggplot2) 
library(dplyr) 
library(factoextra) 
data("BreastCancer", package = "mlbench") 
bc_data <- BreastCancer %>% 
select(-Id) %>% 
na.omit() 
bc_data$Class <- as.factor(bc_data$Class) 
numeric_data <- bc_data %>% 
select(-Class) %>% 
mutate(across(everything(), as.numeric)) 
scaled_data <- scale(numeric_data) 
pca_result <- prcomp(scaled_data, center = TRUE, scale. = TRUE) 
fviz_screeplot(pca_result, addlabels = TRUE, ylim = c(0, 50)) 
fviz_pca_biplot(pca_result, 
geom.ind = "point", 
pointshape = 21, 
pointsize = 2, 
fill.ind = as.factor(bc_data$Class), 
palette = c("#00AFBB", "#FC4E07"), 
addEllipses = TRUE, 
legend.title = "Class") 
summary(pca_result) 



# Load necessary libraries 
library(MASS)  # For LDA 
library(ggplot2)  # For visualization 
# Load the iris dataset 
data(iris) 
# Train the LDA model 
lda_model <- lda(Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, data 
= iris) 
# Print the model summary 
print(lda_model) 
# Predict the class labels using the LDA model 
lda_predictions <- predict(lda_model, iris) 
# Add the predicted values to the original dataset 
iris$lda_pred <- lda_predictions$class 
# Visualize the results using ggplot2 
# Plot the first two components of the LDA (LD1 and LD2) 
lda_df <- data.frame(LD1 = lda_predictions$x[, 1], LD2 = lda_predictions$x[, 2], Species 
= iris$Species) 
ggplot(lda_df, aes(x = LD1, y = LD2, color = Species)) + 
geom_point(size = 3) + 
labs(title = "Linear Discriminant Analysis (LDA) on Iris Dataset", x = "LD1", y = "LD2") 
+ 
theme_minimal()


# Load necessary libraries 
library(ggplot2) 
library(caret) 
# Load the iris dataset 
data(iris) 
# Display the first few rows of the iris dataset 
head(iris) 
# Fit the multiple linear regression model 
# Predicting Sepal.Length based on Sepal.Width, Petal.Length, and Petal.Width 
model <- lm(Sepal.Length ~ Sepal.Width + Petal.Length + Petal.Width, data = iris) 
# Display the summary of the model to analyze coefficients and statistics 
summary(model) 
# Model diagnostic plots (checking residuals) 
par(mfrow = c(2, 2))  # 2x2 grid for plots 
plot(model) 
# Visualize the relationship between the predicted and actual values 
predicted_values <- predict(model, newdata = iris) 
# Scatter plot of actual vs predicted values 
ggplot(iris, aes(x = Sepal.Length, y = predicted_values)) + 
geom_point(aes(color = Species), size = 3) + 
geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") + 
labs(x = "Actual Sepal Length", y = "Predicted Sepal Length", title = "Actual vs Predicted 
Sepal Length") + 
theme_minimal() 
# Checking residuals vs fitted values 
ggplot(data.frame(fitted = fitted(model), residuals = residuals(model)), aes(x = fitted, y = 
residuals)) + 
geom_point(aes(color = residuals), size = 3) + 
geom_hline(yintercept = 0, color = "red") + 
labs(x = "Fitted Values", y = "Residuals", title = "Residuals vs Fitted Values") + 
theme_minimal() 
# Evaluate model accuracy using RMSE (Root Mean Squared Error) 
rmse <- sqrt(mean(residuals(model)^2)) 
cat("Root Mean Squared Error (RMSE):", rmse, "\n") 
# Analyzing Variance Inflation Factor (VIF) for multicollinearity check 
library(car) 
vif(model) 


