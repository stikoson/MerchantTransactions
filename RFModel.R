# Classification model for the iZettle test
# Prepared by Gábor Stikkel
# Prerequisite: R 3.1.2 or higher with the following list of packages installed (see library(...))

library(rjson)
library(plyr)
library(doParallel)
library(randomForest)
library(caTools)
library(ROCR)
library(entropy)

# Utilizing all cores

nodes <- detectCores()
cl <- makeCluster(nodes)
registerDoParallel(cl)

# Loading input files

filename <- "izl_test_merchant_payments.txt"
con <- file(filename, "r")
input <- readLines(con, -1L)
close(con)
payments <- ldply(lapply(input, function(x) t(unlist(fromJSON(x)))))
payments$amount_euro <- as.numeric(payments$amount_euro)

characteristics <- read.csv("izl_test_merchant_characteristics.csv")
labels <- read.csv("izl_test_merchant_labels.csv")

# Setting model parameters

paymentTypes <- names(table(payments$payment_type))
percentile <- 0.25
pLength <- length(seq(0, 1, percentile))

# Feature extraction

featureExtractor <- function(merchantid) {
  merchantData <- payments[payments$merchant_id == merchantid,]
  merchantData <- merchantData[order(merchantData$payment_id),]
  numOfPayments <- nrow(merchantData) 
  
  quantilesPerPaymentType <- c()
  means <- c()
  
  for (ptype in paymentTypes){
    merchantDataPerPaymentType <- merchantData[merchantData$payment_type == ptype,]
    
    quantilesPerPaymentType <- c(quantilesPerPaymentType, 
                            quantile(merchantDataPerPaymentType$amount_euro, seq(0,1,percentile), names=F) )
    means <- c(means, mean(merchantDataPerPaymentType$amount_euro))
    
  }
  quantilesPerPaymentType[is.na(quantilesPerPaymentType)] <- 0
  means[is.na(means)] <- 0
  
  return (c(numOfPayments,
          table(merchantData$payment_type)/numOfPayments,
          quantilesPerPaymentType,
          means
          ))
}

# Distributing feature extraction task to all cores

clusterExport(cl, list("featureExtractor", "payments", "paymentTypes", "percentile"))
merchantFeatures <- ldply(characteristics$merchant_id, featureExtractor, .parallel=T)

# Setting the column names of the feature matrix

X <- laply(seq(0,1,percentile)*100, function(y) laply(paymentTypes, function(x) paste(as.character(x), as.character(y), sep="")))
dim(X) <- c(1, pLength * pLength)
colnames(merchantFeatures) <- c("numOfPayments", #"sumOfPayments", 
                                paymentTypes,
                                X,
                                laply(paymentTypes, function(x) paste(as.character(x), "Means", sep=""))
                                )

# Joining with other tables

merchantFeatures <- cbind(characteristics, merchantFeatures)
merchantFeatures <- join(merchantFeatures, labels, by="merchant_id")
merchantFeatures$merchant_id <- NULL

# Selecting training and test sets

set.seed(123)
split = sample.split(merchantFeatures$class_label, SplitRatio = 0.7)
Train = subset(merchantFeatures, split==TRUE)
Test = subset(merchantFeatures, split==FALSE)

# Training the Random Forest model

RFModel <- randomForest(class_label ~ ., data = Train, ntree = 800)
predictRF <- predict(RFModel, newdata=Test)

# Evaluation

ctable=table(predictRF, Test$class_label)
print("Confusion Matrix:")
print(ctable)
precision = ctable[1,1]/sum(ctable[1,])
recall = ctable[1,1]/sum(ctable[,1])

print("is the accuracy of the model", str(sum(diag(ctable))/nrow(Test)))
print("is the sensitivity/recall of the model", str(recall))
print("is the specificity of the model", str(ctable[2,2]/sum(ctable[,2])))
print("is the precision of the model", str(precision))
print("is the F score of the model", str(2*(precision*recall)/(precision + recall)))

varImpPlot(RFModel)

# Receiver Operating Curve and Area Under the Curve 

prob <- predict(RFModel, newdata=Test, type="prob")[,2]
pred <- prediction(prob, Test$class_label)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")

auc <- performance(pred, measure = "auc")
auc <- auc@y.values[[1]]

roc.data <- data.frame(FalsePositiveRate=unlist(perf@x.values),
                       TruePositiveRate=unlist(perf@y.values),
                       model="RandomForest")

ggplot(roc.data, aes(x=FalsePositiveRate, ymin=0, ymax=TruePositiveRate)) +
  geom_ribbon(alpha=0.2) +
  geom_line(aes(y=TruePositiveRate)) +
  ggtitle(paste0("ROC Curve with AUC=", auc))

