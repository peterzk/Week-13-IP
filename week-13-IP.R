---
  title: "Week 13 IP"
author: "Peter Kiragu"
date: "7/18/2020"
output:
  pdf_document: default
html_document: default
---
  

## 1. Problem Definition

# The goal is to use clustering techniques to understand characteristics of customer groups.
# Compare the effectiveness of two clustering techniques.

### Context 

#Kira Plastinina (Links to an external site.) is a Russian brand that is sold through a defunct chain of retail stores in Russia, Ukraine, Kazakhstan, Belarus, China, Philippines, and Armenia. The brand’s Sales and Marketing team would like to understand their customer’s behavior from data that they have collected over the past year. More specifically, they would like to learn the characteristics of customer groups.

## 2. Data Sourcing



# First we we need to import the dataset

shoppers <- read.csv("online_shoppers_intention.csv")

# Previewing the top of our dataset

head(shoppers)

# Previewing the bottom of our dataset

tail(shoppers)


## 3. Check Data


# Checking the class of our dataset

class(shoppers)

# Our data is stored in a data frame

# Checking dimension of our dataset

dim(shoppers)

# Our dataset has 12,330 rows and 18 columns



# Checking the structure of our dataset

str(shoppers)

# Our dataset has integer values, numerical values, character values and logical values

# Checking the column names

colnames(shoppers)


## 4. Data Cleaning


# Checking for duplicated values 

anyDuplicated(shoppers)

# There are 159 duplicated values that need to be dropped


# Removing the duplicated values 

shoppers <- unique(shoppers)

anyDuplicated(shoppers)

dim(shoppers)


# Checking for null values

colSums(is.na(shoppers))

# There is a total of 96 mission values from 8 columns
# Each of the 8 columns have 12 missing values


# The missing values are insignificant compared to size of the dataset
# For this reason we choose to drop the missing values

shoppers <- na.omit(shoppers)

dim(shoppers)

# After dropping the missing values, the data set has reduced by 12 rows

head(shoppers)


unique(shoppers$VisitorType)


#Extracting the numerical columns from dataset

num <- unlist(lapply(shoppers, is.numeric)) 

numerical_shoppers <- shoppers[ ,num]

head(numerical_shoppers)

dim(numerical_shoppers)


# Getting factors 

shoppers$Month <- as.factor(shoppers$Month)
shoppers$VisitorType <- as.factor(shoppers$VisitorType)
shoppers$Weekend <- as.factor(shoppers$Weekend)
shoppers$Revenue <- as.factor(shoppers$Revenue)


# Extracting categorical columns from dataset

vals <- unlist(lapply(shoppers, is.factor)) 

categorical_shoppers <- shoppers[ ,vals]

head(categorical_shoppers)

dim(categorical_shoppers)


# Checking for outliers 


boxplot(numerical_shoppers, horizontal=FALSE, main="Shoppers Data")


## 5. Exploratory Data Analysis 


# Getting Summary of the data

summary(shoppers)

### 5.1 Univariate EDA


desc_stats <- data.frame(
  Min = apply(numerical_shoppers, 2, min),    # minimum
  Med = apply(numerical_shoppers, 2, median), # median
  Mean = apply(numerical_shoppers, 2, mean),  # mean
  SD = apply(numerical_shoppers, 2, sd),      # Standard deviation
  Max = apply(numerical_shoppers, 2, max),     # Maximum
  IQR = apply(numerical_shoppers, 2, IQR)
)

desc_stats <- round(desc_stats, 1)

head(desc_stats)


# Histograms for numerical columns

par(mfrow=c(2, 2))


for (i in 1:14) {
  hist(numerical_shoppers[, i], main = names(numerical_shoppers)[i], xlab = names(numerical_shoppers)[i])
}


head(categorical_shoppers)


# Barplot for the Month column

month <- table(categorical_shoppers$Month)

barplot(month, main = "Months")


# Barplot for the VisitorType column

visitor_type <- table(categorical_shoppers$VisitorType)

barplot(visitor_type, main = "Visitor Type")

# Barplot for the weekend column

weekend <- table(categorical_shoppers$Weekend)

barplot(weekend, main = "Weekend")


# Barplot for the Revenue column

revenue <- table(categorical_shoppers$Revenue)

barplot(revenue, main = "Revenue")


# Barplot for the Month column

month <- table(categorical_shoppers$Month)

barplot(month)

### 5.2 Bivariate EDA


# Getting a pair plot
plot(numerical_shoppers)


# Plotting distribution of different variables by revenue

library(ggplot2)

ggplot(shoppers, aes(Administrative, colour = Revenue)) +
  geom_freqpoly(binwidth = 1) + labs(title="Administrative Distribution by Revenue")


ggplot(shoppers, aes(Administrative_Duration, colour = Revenue)) +
  geom_freqpoly(binwidth = 1) + labs(title="Administrative Distribution by Revenue")

ggplot(shoppers, aes(ProductRelated, colour = Revenue)) +
  geom_freqpoly(binwidth = 1) + labs(title="Product Related Distribution by Revenue")



ggplot(shoppers, aes(ProductRelated_Duration, colour = Revenue)) +
  geom_freqpoly(binwidth = 1) + labs(title="Administrative Distribution by Revenue")


ggplot(shoppers, aes(BounceRates, colour = Revenue)) +
  geom_freqpoly(binwidth = 1) + labs(title="Bounce_Rate Distribution by Revenue")


# Creating heatmap

library(ggcorrplot)

corr_shoppers <- cor(numerical_shoppers)

ggcorrplot(round(corr_shoppers, 2) ,lab = TRUE ,type = 'lower')

# Dropping unnecessary columns

shoppers2 <- shoppers

drops <- c("ProductRelated_Duration", "ExitRates")

shoppers2 <- shoppers2[, !(names(shoppers2) %in% drops)]

head(shoppers2)



## 6. Feature Engineering


head(shoppers2) # Previewing the dataset



# Label Encoding the categorical variables

library(CatEncoders)


for (i in c(9,14, 15, 16)) {
  
  encode = LabelEncoder.fit(shoppers2[,i])
  shoppers2[,i] = transform(encode, shoppers2[,i])
}

head(shoppers2)


colnames(shoppers2)

# Normalizing the dataset 
normalize <- function(x){
  return ((x-min(x)) / (max(x)-min(x)))
}

shoppers2$Administrative <- normalize(shoppers2$Administrative)
shoppers2$Informational <- normalize(shoppers2$Informational)
shoppers2$ProductRelated <- normalize(shoppers2$ProductRelated)
shoppers2$BounceRates <- normalize(shoppers2$BounceRates)
shoppers2$PageValues <- normalize(shoppers2$PageValues)
shoppers2$SpecialDay <- normalize(shoppers2$SpecialDay)
shoppers2$Administrative_Duration <- normalize(shoppers2$Administrative_Duration)
shoppers2$Informational_Duration <- normalize(shoppers2$Informational_Duration)
shoppers2$Month <- normalize(shoppers2$Month)
shoppers2$Region <- normalize(shoppers2$Region)
shoppers2$TrafficType <- normalize(shoppers2$TrafficType)

head(shoppers2)
dim(shoppers2)


# Reshuffling the data set and creating a uniform distribution

random <- runif(12199)

shoppers2 <- shoppers2[order(random),]

head(shoppers2)



## 7. Implementing Solution 

### 7.1 K-Means Clustering


shoppers2.new <- shoppers2[, c(1:15)]

shoppers2.class <- shoppers2[, "Revenue"]

head(shoppers2.new)
unique(shoppers2.class)


# Implementing clustering 

model_cluster <- kmeans(shoppers2.new, 3)


# Previewing the no. of records in each cluster

model_cluster$size 



# Displaying the cluster centers

model_cluster$centers



table(model_cluster$cluster, shoppers2.class)

# Cluster 1 corresponds to False, cluster 2 corresponds to False and cluster 3 corresponds to False


### 7.1 Hierachical Clustering


# Getting euclidean distance

d <- dist(shoppers2, method = "euclidean")


# Implementing hierarchical clustering

res.hc <- hclust(d, method = "ward.D2")



# Plotting the dedrogram

plot(res.hc, cex = 0.6, hang = -1)

