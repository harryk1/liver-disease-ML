## Libraries used
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot")
if(!require(randomForest)) install.packages("randomForest")

library(tidyverse)
library(data.table)
library(caret)
library(gridExtra)
library(corrplot)
library(randomForest)


## Download csv file from git repo
## csv can be found at path https://github.com/harryk1/liver-disease-ML/tree/main/data
## set temporary working dir to download data
my_dir <- tempdir()
setwd(my_dir)
download.file("https://raw.githubusercontent.com/harryk1/liver-disease-ML/main/data/indian_liver_patient.csv","indian_liver_patient.csv")
# Read data from csv
df <- read_csv("indian_liver_patient.csv")

# Explore data
# Column dataset holds the information if the patient has the disease(value 1) and healthy patient (value 2)
table(df$Dataset)
# Lets convert this to binary data

df <- df %>% mutate(Dataset = Dataset-1)
head(df)

df <- rename(df, Disease = Dataset)
## Convert Disease column as factor vector
df <- df %>% mutate(Disease = as.factor(Disease))
summary(df)
## Convert gender column to factor
df <- df %>% mutate(Gender = as.factor(df$Gender))

## Identify missing data
colSums(sapply(df, is.na))

## Remove 4 NA rows
df <- na.omit(df)

str(df)

## Visualize Data
## Create graphs for each column
age_plot <- df %>% ggplot(aes(Age)) + geom_bar(stat = "count")


g_plot <- df %>% ggplot(aes(Gender, fill=Gender)) + geom_bar(stat = "count")

tb_plot <- df %>% ggplot(aes(Total_Bilirubin)) + 
  geom_histogram(binwidth = 1, color=I("black")) +
  geom_vline(xintercept=mean(df$Total_Bilirubin), color="red")

db_plot <- df %>% ggplot(aes(Direct_Bilirubin)) + 
  geom_histogram(binwidth = 1, color=I("cyan")) +
  geom_vline(xintercept=mean(df$Direct_Bilirubin), color="blue")

ap_plot <- df %>% ggplot(aes(Alkaline_Phosphotase)) + 
  geom_histogram(binwidth = 30, color=I("blue")) +
  geom_vline(xintercept=mean(df$Alkaline_Phosphotase), color="green")

aa_plot <- df %>% ggplot(aes(Alamine_Aminotransferase)) + 
  geom_histogram(binwidth = 30, color=I("red")) +
  geom_vline(xintercept=mean(df$Alamine_Aminotransferase), color="black")

asa_plot <- df %>% ggplot(aes(Aspartate_Aminotransferase)) + 
  geom_histogram(binwidth = 60, color=I("blue")) +
  geom_vline(xintercept=mean(df$Aspartate_Aminotransferase), color="cyan")

tp_plot <- df %>% ggplot(aes(Total_Protiens)) + 
  geom_histogram(binwidth = 0.2, color=I("black")) +
  geom_vline(xintercept=mean(df$Total_Protiens), color="red")

alb_plot <- df %>% ggplot(aes(Albumin)) + 
  geom_histogram(binwidth = 0.2, color=I("cyan")) +
  geom_vline(xintercept=mean(df$Albumin), color="blue")

ag_ratio<- df %>% ggplot(aes(Albumin_and_Globulin_Ratio)) + 
  geom_histogram(binwidth = 0.1, color=I("blue")) +
  geom_vline(xintercept=mean(df$Albumin_and_Globulin_Ratio), color="red")

## Display all graphs in single grid
grid.arrange(age_plot, g_plot, tb_plot, db_plot, ap_plot,
             aa_plot, asa_plot, tp_plot, alb_plot, ag_ratio)

## Create a correlation matrix
factor_columns <- c("Gender", "Disease")

cor_matrix <- cor(df[, !(names(df) %in% factor_columns)])
cor_matrix
## Visualize Correlations
corrplot(cor_matrix)

## Identify high correlation pairs, so that we can drop the columns for a better feature selection
## find columns with correlation > 0.75
high_corr_cols <- findCorrelation(cor_matrix, cutoff = 0.75, names = TRUE)
high_corr_cols


## Remove high correlated columns
df <- df[, !(names(df) %in% high_corr_cols)]
head(df)

###
## Split data into 70% train_set and 30% test_set
set.seed(1, sample.kind = "Rounding")

test_index <-createDataPartition(df$Disease, p=0.3,list=FALSE)
train_set <-df[-test_index,]
test_set <- df[test_index,]
table(train_set$Disease)
table(test_set$Disease)


## Predictive Analysis using Random Forest
set.seed(14, sample.kind = "Rounding")    # simulate R 3.5
train_rf <- train(Disease ~ .,
                  data = train_set,
                  method = "rf",
                  ntree = 100,
                  tuneGrid = data.frame(mtry = seq(1:7)))

train_rf$bestTune
ggplot(train_rf)

accuracy_rf <- confusionMatrix(predict(train_rf, test_set,
                        type = "raw"),
                test_set$Disease)$overall["Accuracy"]
rf_preds <- predict(train_rf, test_set)
mean(rf_preds == test_set$Disease)
## Get rf vriables by importance
imp <- varImp(train_rf)
imp

## Train GLM by all predictors
set.seed(1, sample.kind = "Rounding")

train_glm <- train(Disease ~ . , method = "glm", data = train_set)

pred_glm <- predict(train_glm, test_set)

accuracy_glm <- confusionMatrix(data = pred_glm, reference = test_set$Disease)$overall["Accuracy"]


## Train GLM on reduced features
set.seed(1, sample.kind = "Rounding")

train_glm1 <- train(Disease ~ . - Gender - Albumin_and_Globulin_Ratio , method = "glm", data = train_set)

pred_glm1 <- predict(train_glm1, test_set)

accuracy_glm1 <- confusionMatrix(data = pred_glm1, reference = test_set$Disease)$overall["Accuracy"]

## Using Knn with control parameter
## tune knn model
set.seed(6, sample.kind = "Rounding")
control <- trainControl(method = "cv", number = 10, p = .9)
train_knn <- train(Disease ~ ., method = "knn",
                      data = train_set,
                      tuneGrid = data.frame(k = seq(3,71,2)),
                      trControl = control)
ggplot(train_knn, highlight = TRUE)

# We get highest accuracy when k=35
train_knn$bestTune

## Plot accuracy graph
train_knn$results %>%
  ggplot(aes(x = k, y = Accuracy)) +
  geom_line() +
  geom_point() +
  geom_errorbar(aes(x = k,
                    ymin = Accuracy - AccuracySD,
                    ymax = Accuracy + AccuracySD))

pred_knn <- predict(train_knn, test_set, type = "raw")
accuracy_knn <- confusionMatrix(pred_knn, test_set$Disease)$overall["Accuracy"]
accuracy_knn

rm(accuracy_results)
accuracy_results <- tibble(method = "Random Forest",
                           Accuracy = accuracy_rf )
accuracy_results <- bind_rows(accuracy_results, tibble(method = "GLM",
                                                       Accuracy = accuracy_glm ))
accuracy_results <- bind_rows(accuracy_results, tibble(method = "GLM - Reduced features",
                                                       Accuracy = accuracy_glm1 ))
accuracy_results <- bind_rows(accuracy_results, tibble(method = "KNN",
                                                       Accuracy = accuracy_knn ))
accuracy_results %>% knitr::kable()

## GLM with reduced features has highest accuracy of 72% to detect liver disease in a person with the given data
