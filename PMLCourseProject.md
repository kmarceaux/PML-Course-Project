---
title: "PML Course Project"
author: "K. Marceaux"
date: "October 3, 2018"
output: 
  html_document: 
    fig.path: P:/COURSERA
    fig_caption: yes
    keep_md: yes
---
###Introduction and Objective
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.  

The goal is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants who were asked to perform barbell lifts correctly and incorrectly in 5 different ways.  Ultimately, we are trying to predict the manner in which they did the exercise. This is the "classe" variable in the training set. 

Information on the dataset can be found here:  [Lifting Exercise Dataset](http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har)  


###Packages Required and Reproducibility 
Load packages needed and set seed for reproducibility. 



```r
library(readxl)
library(data.table)
library(rpart)
library(rpart.plot)
library(caret) 
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(randomForest)
```

```
## randomForest 4.6-14
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
set.seed(0618)
```


###Data
Load training and test data sets.  Summaries not shown due to length of the output. 

*Download and read the data.*  


```r
## Download training dataset 
if(!file.exists("./RRdatasets")){dir.create("./RRdatasets")}
fileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(fileUrl,destfile="./RRdatasets/training.csv")


trainingdata <- read.csv("./RRdatasets//training.csv", header = TRUE,
                           sep = ",", na.strings=c("NA", "DIV/0", "")) 

##Download testing dataset
if(!file.exists("./RRdatasets")){dir.create("./RRdatasets")}
fileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(fileUrl,destfile="./RRdatasets/testing.csv")


testingdata <- read.csv("./RRdatasets//testing.csv", header = TRUE,
                           sep = ",", na.strings=c("NA", "DIV/0", "")) 
```


```r
head(trainingdata)
head(testingdata)

summary(trainingdata)
summary(testingdata)
```
####Clean and Transform
Cleaning the dataset of variables with no predictive value was done (X, user_name, plus 5 more).  Variables with missing values were also removed.



```r
###Removed X, user_name, timestamps, etc. as these data elements are not useful for this project. 

trainingdata <- trainingdata[-c(1:7)]
testingdata <- testingdata[-c(1:7)]

###Removed data with missing values

trainingdata<- trainingdata[, colSums(is.na(trainingdata))==0]
testingdata<- testingdata[, colSums(is.na(testingdata))==0]


####Datasets observations and variables
dim(trainingdata)
```

```
## [1] 19622    53
```

```r
dim(testingdata)
```

```
## [1] 20 53
```
####Partition the data
To perform cross validation, the training set must be partitioned.  This was done with a 60/40 split.  (60% for training and 40% for testing)
 

```r
inTrain = createDataPartition(trainingdata$class, p = 0.6)[[1]]
PMLtraining = trainingdata[inTrain,]
PMLtesting = trainingdata[-inTrain,]

dim(PMLtraining)
```

```
## [1] 11776    53
```

```r
dim(PMLtesting)
```

```
## [1] 7846   53
```

###Model Development 
A common machine learning method is known as "predicting with trees".  Using decision trees one starts with all the variables then find the first variable that best splits the outcome into two groups.  You then divide the data into two more groups and split and on and on until the groups are too small or they are sufficiently homogeneous.  As the goal is to predict the manner in which a participant did an exercise, the decision tree method seemed a logical place to start with a prediction model.  

Per the lecture in the Practical Machine Learning class, random forests is similar to bagging in the sense that we bootstrap samples, so we take a resample of our observed data, and our training data set. And then we rebuild classification or regression trees on each of those bootstrap samples.  The idea is to grow a large number of trees.  

####Data Exploration
The "classe" variable, which we are trying to predict, has 5 levels. Below is the frequency of each classe level within the training dataset. 


```r
plot(PMLtraining$classe, col="black", main="Frequency of Classe variable in Training Dataset", xlab="classe levels", ylab="Frequency")
```

![](PMLCourseProject_files/figure-html/explore-1.png)<!-- -->

####Prediction Model Using Decision Tree (with results)


```r
modelDT <- rpart(classe ~., data=PMLtraining, method="class")
predDT <- predict(modelDT, PMLtesting, type="class")
confusionMatrix(PMLtesting$classe, predDT)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2096   55   41   31    9
##          B  342  756  298   85   37
##          C   35   71 1162   98    2
##          D  130   32  130  891  103
##          E   42   69  173   94 1064
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7608          
##                  95% CI : (0.7512, 0.7702)
##     No Information Rate : 0.3371          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6957          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.7924  0.76907   0.6441   0.7431   0.8757
## Specificity            0.9739  0.88897   0.9659   0.9406   0.9430
## Pos Pred Value         0.9391  0.49802   0.8494   0.6928   0.7379
## Neg Pred Value         0.9022  0.96413   0.9009   0.9530   0.9764
## Prevalence             0.3371  0.12529   0.2299   0.1528   0.1549
## Detection Rate         0.2671  0.09635   0.1481   0.1136   0.1356
## Detection Prevalence   0.2845  0.19347   0.1744   0.1639   0.1838
## Balanced Accuracy      0.8831  0.82902   0.8050   0.8418   0.9094
```

```r
###Plot of Decition Tree
rpart.plot(modelDT, main="Decision Tree", type=2, extra=102)
```

![](PMLCourseProject_files/figure-html/DT model-1.png)<!-- -->

####Prediction Model Using Random Forests (with results)

```r
modelRF <- train(classe ~ ., method = "rf", data=PMLtraining)
predRF <- predict(modelRF, PMLtesting)
confusionMatrix(PMLtesting$classe, predRF)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    0    0    0    0
##          B   13 1498    7    0    0
##          C    0    8 1352    8    0
##          D    0    0   17 1269    0
##          E    0    2    1    1 1438
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9927          
##                  95% CI : (0.9906, 0.9945)
##     No Information Rate : 0.2861          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9908          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9942   0.9934   0.9818   0.9930   1.0000
## Specificity            1.0000   0.9968   0.9975   0.9974   0.9994
## Pos Pred Value         1.0000   0.9868   0.9883   0.9868   0.9972
## Neg Pred Value         0.9977   0.9984   0.9961   0.9986   1.0000
## Prevalence             0.2861   0.1922   0.1755   0.1629   0.1833
## Detection Rate         0.2845   0.1909   0.1723   0.1617   0.1833
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9971   0.9951   0.9897   0.9952   0.9997
```
####Results
Per the models, the Random Forest algorithm performed better than the Decision Tree algorithm.  Per the classification matrices, the accuracy of the Decision Tree Model is 0.7608 while the accuracy of the Random Forest model is 0.9921.  Thus, the random forest model is chosen.  The test data has 20 cases, so with an accuracy of 99% on the cross-validation data, we can expect to have few (or no) of the test samples missclassified.  

####Predict outcomes of Test dataset
The Random Forest model was applied to the test data.  Here are the results. 


```r
predictclasse <- predict(modelRF, testingdata)
predictclasse
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```



 
 

