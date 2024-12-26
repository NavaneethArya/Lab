#program 1

df1 <- data.frame(ID=1:5, Name=c("a","b","c","d","e"))
df2 <- data.frame(ID=3:7, Age=c(12,23,34,45,67))

merged_df <- merge(df1,df2,by="ID", all=TRUE)
print(merged_df)

library(tidyr)
long <- gather(merged_df,"Variable","Value", -ID)
print(long)

wide <- spread(long,"Variable", "Value")
print(wide)

#program 2

strings <- c("Data_transformation", "computational_statisticts")
library(stringr)
strings <- str_replace_all(strings,"_"," ")
print(strings)

emails <- c("user1@gmail.com", "user2@example.com")
valid_emails <- grep("@gmail.com",emails,value=TRUE)
print(valid_emails)

#program 3

library(dplyr)
library(forecast)
set.seed(123)
data <- data.frame(
  Date=seq(as.Date("2020-01-01"),by="month", length.out=36),
  value1=rnorm(36,mean=200,sd=20),
  value2=rnorm(36,mean=100,sd=10)
)

data_ts <- ts(data[,-1],start = c(2020,1), frequency = 36)

grouped_data <- data %>%
  mutate(Year=format(Date, "%Y")) %>%
  group_by(Year)%>%
  summarise(across(value1:value2,mean))

print(data)
print(grouped_data)

fit1 <- auto.arima(data_ts[,1])
fit2 <- auto.arima(data_ts[,2])
forecast1 <- forecast(fit1,h=6)
forecast2 <- forecast(fit2,h=6)

par(mfrow=c(2,1))
plot(forecast1)
plot(forecast2)

print(forecast1)
print(forecast2)

checkresiduals(fit1)


# program 4

data <- c(10, 10, 30, 40, 10, 60, 70, 80, 90, 100)


results <- list(
  mean = mean(data),
  median = median(data),
  mode = as.numeric(names(sort(table(data), decreasing = TRUE)[1])), 
  sd = sd(data),
  variance = var(data),
  MAD = mad(data), 
  quartile_dev = IQR(data) / 2 
)

print(results)

#program 5

library(caret)
library(Metrics)

data(mtcars)

trainIndex <- createDataPartition(mtcars$mpg, p=0.8, list=FALSE)
trainData <- mtcars[trainIndex,]
validData <- mtcars[-trainIndex,]

calc_metrics <- function(actual,predicted){
  c(
    RMSE <- rmse(actual,predicted),
    MAE <- mae(actual,predicted),
    R2 <- cor(actual,predicted)^2
  )
}

model_val <- lm(mpg ~ ., data=trainData)
metrics_val <- calc_metrics(validData$mpg, predict(model_val, newdata = validData))
print(metrics_val)

metrics_loocv <- sapply(1:nrow(mtcars), function(i){
  model_loocv <- lm(mpg ~ ., data=mtcars[-i,])
  pred_loocv <- predict(model_loocv, newdata=mtcars[i, , drop=FALSE])
  calc_metrics(mtcars$mpg[i],pred_loocv)
})
print(colMeans(metrics_loocv))

cv_results <- train(mpg ~ ., data=mtcars, method="lm",
                    trControl=trainControl(method="cv", number = 5, summaryFunction = defaultSummary))
print(cv_results$results)

#program 6

library(ggplot2) 
library(dplyr) 

set.seed(123) 

response_data <- data.frame( 
  response = c("Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"), 
  frequency = c(5, 15, 25, 30, 25) 
) 

print("Frequency Distribution:") 
print(response_data) 

response_data <- response_data %>% 
  mutate(proportion = frequency / sum(frequency)) 

mu <- mean(1:length(response_data$response)) 
sigma <- sd(1:length(response_data$response)) 
n <- sum(response_data$frequency)  
p <- mean(response_data$proportion) 
lambda <- n * p 
p_bernoulli <- response_data$proportion[1]  

x <- 0:(max(response_data$frequency) + 10) 

normal_df <- data.frame(x = x, density = dnorm(x, mean = mu, sd = sigma)) 
binomial_df <- data.frame(x = x, density = dbinom(x, size = n, prob = p)) 
poisson_df <- data.frame(x = x, density = dpois(x, lambda = lambda)) 
bernoulli_df <- data.frame(x = c(0, 1), density = c(1 - p_bernoulli, p_bernoulli)) 

ggplot() + 
  geom_line(data = normal_df, aes(x = x, y = density), color = "blue") + 
  geom_line(data = binomial_df, aes(x = x, y = density), color = "red") + 
  geom_line(data = poisson_df, aes(x = x, y = density), color = "green") + 
  geom_bar(data = bernoulli_df, aes(x = factor(x), y = density), stat = "identity", fill = "purple", alpha = 0.5) + 
  labs(title = "Probability Distributions", 
       x = "Response Categories / Counts", 
       y = "Density / Probability") + 
  scale_x_discrete(labels = c("Failure", "Success")) + 
  theme_minimal() + 
  theme(legend.position = "top") + 
  scale_color_manual(values = c("blue", "red", "green")) 

cat("Summary Statistics for the Frequency Distribution:\n") 
cat("Mean of Frequencies: ", mean(response_data$frequency), "\n") 
cat("Standard Deviation of Frequencies: ", sd(response_data$frequency), "\n") 
cat("Total Responses: ", sum(response_data$frequency), "\n") 
cat("Probability of 'Strongly Disagree': ", p_bernoulli, "\n") 
cat("Mean for Normal Distribution (mu): ", mu, "\n") 
cat("Standard Deviation for Normal Distribution (sigma): ", sigma, "\n") 
cat("Lambda for Poisson Distribution: ", lambda, "\n") 


#program 7

set.seed(123)

one_sample_data <- rnorm(30, mean=50, sd=10)

two_sample_data1 <- rnorm(30, mean=55, sd=10)
two_sample_data2 <- rnorm(30, mean=50, sd=10)

paired_sample_before <- rnorm(30, mean=50, sd=10)
paired_sample_after <- paired_sample_before+rnorm(30, mean=-2, sd=5)

one_sample_test <- t.test(one_sample_data,mu=50)
print(one_sample_test)

two_sample_test <- t.test(two_sample_data1, two_sample_data2)
print(two_sample_test)

paired_sample_test <- t.test(paired_sample_before, paired_sample_after, paired=TRUE)
print(paired_sample_test)

#program 8

library(dplyr)
set.seed(123)

treatment_A <- c(11,22,33,44,55)
treatment_B <- c(22,33,44,55,66)
treatment_C <- c(33,44,55,66,77)
one_way <- data.frame(
  value=c(treatment_A, treatment_B, treatment_C),
  treatment = factor(rep(c("A","B","C"),each=5))
)

one_way_anova <- aov(value~treatment,data=one_way)
print(summary(one_way_anova))

treatment_A_male <- c(11,22,33,44,55)
treatment_A_female <- c(22,33,44,55,66)
treatment_B_male <- c(11,22,33,44,55)
treatment_B_female <- c(22,33,44,55,66)

two_way <- data.frame(
  value = c(treatment_A_male, 
             treatment_A_female,
             treatment_B_male,
             treatment_B_female),
  treatment = factor(rep(c("A","B"),each=10)),
  gender = factor(rep(c("male","female"),times=10))
)

two_way_anova <- aov(value ~ treatment * gender, data=two_way)
print(summary(two_way_anova))

#program 9

library(dplyr)
library(reshape2)
library(corrplot)
library(ggplot2)

data(mtcars)

correlation_matrix <- cor(mtcars)
print(correlation_matrix)

rank_correlation_matrix <- cor(mtcars, method="spearman")
print(rank_correlation_matrix)

model <- lm(mpg ~ hp+wt, data=mtcars)
print(summary(model))

mtcars <- mtcars %>%
  mutate(predicted_mpg=predict(model),
         residuals=residuals(model))

ggplot(mtcars, aes(x=predicted_mpg, y=mpg))+
  geom_point(color='blue')+
  geom_smooth(method = 'lm', color='red')+
  labs(title='actual vs predicted mpg', x='predicted', y='actual')+
  theme_minimal()

cor_melted <- melt(correlation_matrix)
ggplot(cor_melted,aes(Var1,Var2,fill=value))+
  geom_tile()+
  scale_fill_gradient2(low='blue', high='red', mid='white', limit=c(-1,1))+
  labs(title = "heatmap")+
  theme_minimal()

corrplot(correlation_matrix, method="circle", type='upper')


#program 10

library(mlbench)
library(ggplot2)
library(dplyr)
library(factoextra)

data("BreastCancer", package = "mlbench")
bc_data <- na.omit(BreastCancer %>% select(-Id))
bc_data$Class <- as.factor(bc_data$Class)

scaled_data <- scale(bc_data %>% select(-Class) %>% mutate(across(everything(), as.numeric)))
pca_result <- prcomp(scaled_data, center = TRUE, scale. = TRUE)

fviz_screeplot(pca_result, addlabels = TRUE, ylim = c(0, 50))
fviz_pca_biplot(pca_result, geom.ind = "point", pointshape = 21, pointsize = 2, 
                fill.ind = bc_data$Class, palette = c("#00AFBB", "#FC4E07"), 
                addEllipses = TRUE, legend.title = "Class")
summary(pca_result)


#program 11

library(ggplot2)
library(MASS)

data(iris)
lda_model <- lda(Species ~ Sepal.Length+Sepal.Width+Petal.Length+Petal.Width, data=iris)
print(lda_model)

lda_predictions <- predict(lda_model,iris)

iris$lda_pred <- lda_predictions$class

lda_df <- data.frame(LDA1=lda_predictions$x[,1], LDA2=lda_predictions$x[,2], Species=iris$Species)

ggplot(lda_df,aes(x=LDA1, y=LDA2, color=Species))+
  geom_point(size=3)+
  labs(title="linear discriminant analysis", x="LDA1", y="LDA2")+
  theme_minimal()


#program 12

library(ggplot2)
library(caret)

data(iris)
head(iris)

model <- lm(Sepal.Length ~ Sepal.Width+Petal.Length+Petal.Width, data=iris)
summary(model)

par(mfrow=c(2,2))
plot(model)

predicted_values <- predict(model,newdata = iris)
ggplot(iris,aes(x=Sepal.Length,y=predicted_values,color=Species))+
  geom_point(size=3)+
  geom_abline(intercept = 0, slope=1, linetype="dashed", color="red")+
  labs(x="actual sepal length", y="predicted sepal length")+
  theme_minimal()

ggplot(data.frame(fitted=fitted(model),
                  residuals=residuals(model)),
       aes(x=fitted, y= residuals))+
  geom_point(size=3)+
  geom_hline(yintercept = 0,color="red")+
  labs(x="fitted",y="residuals")+
  theme_minimal()

cat(" root mean square error (RMSE) :", sqrt(mean(residuals(model)^2)) )

vif(model)
