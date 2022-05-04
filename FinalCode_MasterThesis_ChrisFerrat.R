#####################################################################################
# Main Code for Main Part of Master Thesis - Model Averaging of Binary Response - Chris Ferrat
#####################################################################################

#Table of Contents
# 2.4 Theoretical Basics of ROC
# 4. Application in R
# 4.1 Logistic Regression with Bankdata
# 4.2 Class Prediction with Logistic Regression
# 4.3 Model Averaging of Binary Response
# 4.4 Class Prediction of Averaged Model
# 4.5 Output of Both Models
# 4.6 Comparison of Confusion Matrices
# 4.7 Comparison of ROC
# 4.8 The Idea of Exact Deviations
# 4.9 Learning Curve
# 5. Selected Examples of Other Datasets
# 5.1 GermanCredit
# 5.2 AutoClaim
# 5.3 Wine

#For CM in the thesis: Run the help functions first to plot the figures!

#To keep it simple, just install all packages and run step by step through the code.
#Possible Problems:
#Problem 1: If your PC fails to predict with the averaged model after running the whole code, restart R
#and run the code again BUT skip part "4.3 MODEL INSPECTION WITH DIFFERENT COMMANDS" after defining
#the variables of the averaged model and library(mumin), then run "4.4 Class Prediction of Averaged Model"
#Problem 2: If your PC fails during the loop after running the whole code, restart R,
#load the data again and directly run the loops+code after the loops
#Further Datasets: Same procedure

rm(list = ls())
WorkDir <- ("D:/Dropbox/Bearbeitete Dokumente Uni Basel/22Masterarbeit/Code")
setwd(WorkDir)
options(mc.cores = parallel::detectCores())

##### Necessary packages for Main Part #####
install.packages("caTools")
install.packages("lattice")
install.packages("ggplot2")
install.packages("caret")
install.packages("MuMIn")
install.packages("texreg")
install.packages("tictoc")
install.packages("ROCR")
install.packages("tidyverse")
install.packages("cplm")
install.packages("graphics")
install.packages("jtools")
install.packages("kableExtra")
install.packages("ggstance")
install.packages("broom.mixed")
install.packages("huxtable")


##### Packages for additional applications and checking results in the Main Part #####
install.packages("modelsummary")
install.packages("gmodels")
#install.packages("arm") #not used in this thesis

#Check package versions
#packageVersion("huxtable")

#####################################################################################
############# 2.4 Figures for conceptual basics - Example ROC #######################
#####################################################################################

xaxis <- c(0,0,0.02,0.02,0.05,0.05,0.12,0.12,0.16,0.16,0.2,0.2,0.34,0.34,0.4,0.4,0.45,0.45,0.5,0.5,0.55,0.55,0.6,0.6,0.7,0.7,0.82,0.82,0.88,0.88,0.94,0.94,1)
yaxis <- c(0,0,0,0.1,0.1,0.2,0.2,0.4,0.4,0.56,0.56,0.65,0.65,0.7,0.7,0.74,0.74,0.76,0.76,0.79,0.79,0.82,0.82,0.83,0.83, 0.87,0.87,0.9,0.9,0.96,0.96,1,1)
ExampleROC <- data.frame(cbind(xaxis,yaxis))
plot(ExampleROC$xaxis,ExampleROC$yaxis, main = "ROC Example" ,xlab = "False Positive Rate (1-Specifity)", ylab = "True Positive Rate (Sensitivity)", type = "o", xlim = c(0,1), ylim = c(0,1), col='blue', lwd = 6)

#grey area under the curve
#Dropping end points to x-axis
Polyx = c(1, xaxis, 1)
Polyy = c(0, yaxis, 0)
## plot polygon between curve and x-axis
polygon(Polyx,Polyy,col="lightgrey")
#Useless ROC similar to guessing 50/50
abline(a=0, b = 1, col='orange', lwd = 6)
#Plot the blue line (example ROC) again
lines(ExampleROC$xaxis, ExampleROC$yaxis, col="blue", type = "o", lwd = 6)
#Legend
legend(0.5,0.29, 
       c("Example ROC","Guessing 50/50","AUC - Area under the curve"), 
       lty=c(1,1,1), 
       lwd=c(4,4,4),
       col=c("blue","orange","grey"))

#####################################################################################
################# 2.4 New clean plot for Extreme Cases ROC ##########################
#####################################################################################

plot(ExampleROC$xaxis,ExampleROC$yaxis, main = "ROC Extreme Cases" ,xlab = "False Positive Rate (1-Specifity)", ylab = "True Positive Rate (Sensitivity)", type = "o", xlim = c(0,1), ylim = c(0,1), col='white', lwd = 6)
#Useless ROC similar to guessing 50/50
abline(a=0, b = 1, col='orange', lwd = 6)
#perfect ROC - save points manually and draw a line
pRx <- c(0,0,1)
pRy <- c(0,1,1)
lines(pRx, pRy, col="darkblue",lwd=6, type="l")
#perfectly wrong ROC where guessing would be better
pfRx <- c(0,1,1)
pfRy <- c(0,0,1)
lines(pfRx, pfRy, col="red",lwd=6, type="l")

#Legend
legend(0.6,0.29, 
       c("Perfect ROC","Guessing 50/50","Perfectly wrong ROC"), 
       lty=c(1,1,1), 
       lwd=c(4,4,4),
       col=c("darkblue","orange","red"))

#####################################################################################
############ 2.4 2 Overlapping Distributions for Theoretical Basics ##################
#####################################################################################

x <- seq(-10, 10, length=10000)
dist1 <- dnorm(x, mean=0.35, sd=0.12)
dist2 <- dnorm(x, mean=0.65, sd=0.12)
#plot first distribution
plot(x, dist1, yaxt='n', type="l", lwd=4, xlim = c(0,1), ylim = c(0,3.5), col='blue', main= "Distribution Separation", xlab = "Threshold 0.5", ylab = "Density of Classes", xaxs="i",yaxs="i")
abline(v=0.5, col='black', lwd = 3)
#plot second distribution
lines(x, dist2, col="red",lwd=4, type="l")
#Insert text
text(0.35, 2, "TN", cex = 2)
text(0.65, 2, "TP", cex = 2)
text(0.45, 0.3, "FN", cex = 2)
text(0.55, 0.3, "FP", cex = 2)

legend("topleft", 
       c("Negative Class - 0","Positive Class - 1"), 
       lty=c(1,1), 
       lwd=c(4,4),
       col=c("blue","red"))

#####################################################################################
############### 4. MAIN PART OF MASTER THESIS WITH BANKDATA #########################
#####################################################################################
rm(list = ls())

WorkDir <- ("D:/Dropbox/Bearbeitete Dokumente Uni Basel/22Masterarbeit/Code")
setwd(WorkDir)
options(mc.cores = parallel::detectCores())

#####################################################################################
######################## 4. Load and inspect Dataset #################################
#####################################################################################

#Download dataset from this website: http://archive.ics.uci.edu/ml/datasets/Bank+Marketing
#then click on "Download: Data Folder" which is topleft and chose "bank.zip" on the following website
#After unpacking the ZIP in a folder called "Bankdata" on your harddrive, take subset of this dataset called "bank.csv", not bankfull
bankdata <- read.csv2("D:/Dropbox/Bearbeitete Dokumente Uni Basel/22Masterarbeit/Bankdata/bank.csv", header = TRUE)
table(bankdata$y)#y = subscribe
#no: 4000, yes: 521

#Create Binary for y
bankdata$subscribed <- ifelse(bankdata$y == 'yes', 1, 0)#create binary variable: 1=subscribed, 0=no subscription
bankdatauncleaned <- bankdata[,c(18,17,1:16)] #new order
bankdatauncleaned <- bankdatauncleaned[,c(1,3:11,15,17,18)]#Create subset with chosen variables without variable 2,12,13,14,16
table(bankdatauncleaned$subscribed)
bankdata <- na.omit(bankdatauncleaned) #cleaning the dataset from NA variables to get just complete observations
table(bankdata$subscribed)

#Change character variables into factors
summary(bankdata)
bankdata$job <- as.factor(bankdata$job)
bankdata$marital <- as.factor(bankdata$marital)
bankdata$education <- as.factor(bankdata$education)
bankdata$default <- as.factor(bankdata$default)
bankdata$housing <- as.factor(bankdata$housing)
bankdata$loan <- as.factor(bankdata$loan)
bankdata$contact <- as.factor(bankdata$contact)
bankdata$poutcome <- as.factor(bankdata$poutcome)
summary(bankdata)
bankdata <- na.omit(bankdata)#to be sure, clean dataset again
table(bankdata$subscribed)

#####################################################################################
### 4. Split dataset - 70% train data (Insample) und 30% test data (Out-of-sample) ##
#####################################################################################
library(caTools)
set.seed(125)#choose seed
sample = sample.split(bankdata$subscribed, SplitRatio = (.7))
banktrainfull = subset(bankdata, sample == TRUE)
banktest  = subset(bankdata, sample == FALSE)

#####################################################################################
########################### 4.1 Create Logit model ##################################
#####################################################################################
#logit regression with train data
banklogittraindata.fit = glm(subscribed ~ age + job + marital + education + default + balance + housing + loan + contact
                              + campaign + previous + poutcome, control=glm.control(maxit=100), data=banktrainfull, family=binomial)

#Output and Coefficients
summary(banklogittraindata.fit)

library(jtools)
library(kableExtra)
library(ggstance)
library(broom.mixed)
library(texreg)
#Nice summary => clear overview
summ(banklogittraindata.fit, digits = 4)

#Additional ways to show proper output
screenreg(banklogittraindata.fit, single.row = TRUE, custom.model.names = c("Logit Model")) #used for comparison later

#Additional ways to show coefficients
#library(huxtable)
#huxtablereg(banklogittraindata.fit)

#export_summs(banklogittraindata.fit, scale = TRUE) #for combining models in 1 table
plot_summs(banklogittraindata.fit, scale = TRUE) #coefficient comparison (average model not compatible!)

#Additional
#try and error
#library(modelsummary)
#models <- list()
#models[['Logit']] <- banklogittraindata.fit
#modelsummary(models)

#####################################################################################
###################### 4.2 Perform logit.fit on train data ##########################
#####################################################################################
banklogitfittraindata.prob = predict(banklogittraindata.fit, banktrainfull, type="response")

#Round and shape the predicted probabilities
ROUNDbanklogitfittraindata.prob <- round(banklogitfittraindata.prob, digits = 3)
banktrainpred <- ifelse(ROUNDbanklogitfittraindata.prob >= "0.5", 1, 0)
fbanktrainpred <- factor(banktrainpred)
levels(fbanktrainpred) = c('NoSubscription','Subscription')

#Prepare the real outcomes
banktrainreal <- banktrainfull[,1]
fbanktrainreal <- factor(banktrainreal)
levels(fbanktrainreal) = c('NoSubscription','Subscription')

#Creating confusion matrix - how performs the model on the train data - Insample
library(lattice)
library(ggplot2)
library(caret)
ConfMattrain <- confusionMatrix(data=fbanktrainpred, reference = fbanktrainreal)#Reference = Real data

#Display results in simple form
ConfMattrain

#Plot for short overview
fourfoldplot(ConfMattrain$table, main = "Logit Traindata")

#USE FUNCTION "draw_confusion_matrix_bank"! Execute all functions for later use!
#Plot Confusion Matrix with function "draw_confusion_matrix":
draw_confusion_matrix_bank_train(ConfMattrain)

############## Additional: Check if the same results (not in the thesis itself!)
library(gmodels)
#Computes the crosstable calculations
CrossTable(fbanktrainpred,fbanktrainreal)
#Double check if same numbers
table(fbanktrainpred)
table(fbanktrainreal)

#####################################################################################
##################### 4.2 Perform logit.fit on test data ############################
#####################################################################################
#Perform logit.fit from train data on test data
banklogitfittestdata.prob = predict(banklogittraindata.fit, banktest, type="response")

#Round and prepare predicted probabilities
ROUNDbanklogitfittestdata.prob <- round(banklogitfittestdata.prob, digits = 3)
banktestpred <- ifelse(ROUNDbanklogitfittestdata.prob >= "0.5", 1, 0)
fbanktestpred <- factor(banktestpred)
levels(fbanktestpred) = c('NoSubscription','Subscription')

banktestreal <- banktest[,1]
fbanktestreal <- factor(banktestreal)
levels(fbanktestreal) = c('NoSubscription','Subscription')

#Creating confusion matrix - how performs the model on the test data - Out-of-sample
ConfMattest <- confusionMatrix(data=fbanktestpred, reference = fbanktestreal)

#Display results 
ConfMattest

#Plot
fourfoldplot(ConfMattest$table, main = "Logit Testdata")

#Plot Confusion Matrix with function "draw_confusion_matrix - execute all if not already done!
draw_confusion_matrix_bank_test(ConfMattest)

######## Check if same results (not in the thesis!)
library(gmodels)
#Computes the crosstable calculations
CrossTable(fbanktestpred,fbanktestreal)
#Check if correct numbers
table(fbanktestpred)
table(fbanktestreal)

#####################################################################################
# 4.2 Compare Confusion Matrix of Traindata and Testdata - Logit Insample VS Out-Of-Sample
#####################################################################################
#Exact
draw_confusion_matrix_bank_train(ConfMattrain)
draw_confusion_matrix_bank_test(ConfMattest)
#Plot
par(mfrow=c(1,2)) #2 graphs side by side
fourfoldplot(ConfMattrain$table, main = "Logit Traindata")
fourfoldplot(ConfMattest$table, main = "Logit Testdata")


#####################################################################################
##################### 4.3 Perform Average Method ####################################
#####################################################################################
library(MuMIn)
bank1 <- glm(subscribed ~ age + job + marital + education + default + balance + housing + loan + contact
             + campaign + previous + poutcome, data=banktrainfull, family=binomial, control=glm.control(maxit=100), na.action = na.fail)

#standardize coefficients if you want to interpret them (NOT USED FOR THE THESIS, not working in this form)
#library(MASS)
#library(Matrix)
#library(lme4)
#library(arm)
#bank1 <- standardize(bank1, standardize.y = FALSE)

#####################################################################################
############### 4.3 MODEL INSPECTION WITH DIFFERENT COMMANDS #########################
#####################################################################################
#Normal fitting, but cant use it for predict function in this form
#model selection: you can combine models manually(model.sel) or automatically over "dredge"
(ms1 <- dredge(bank1, rank = "AICc", extra = c("R^2","adjR^2", F = function(x) #Create Model Selection table
  summary(x)$fstatistik[[1]])))
#write.csv(ms1,"D:\\Dropbox\\Bearbeitete Dokumente Uni Basel\\22Masterarbeit\\Schriftliche Arbeit\\ModelSelection1PresentationMuMinCommands.csv", row.names = FALSE)

ms1 #Show model selection table
ms1[1:4] #show model selection table but new weights!

#Subset best 5 models
ms1Subset <- ms1[1:5, ]
screenreg(ms1Subset, custom.model.names = c("Best Model", "2. Best Model", "3. Best Model", "4. Best Model", "5. Best Model"))#show best 5 models

#Plot a graph of all models and their weights
par(mfrow=c(1,1)) #Plot just 1 graph
par(oma = c(0,3,3,3)) #increase distance to window borders to prevent cutted plot
plot(ms1) #Make plot window much bigger to plot without cutting
#If you plot it without par you get the axis description, if your lucky the plot is also not cutted

#After creating a model selection table, you can use "model.avg" to choose a subset of models
ms11 <- model.avg(ms1, subset = delta < 1.5) #Averaging the models with subset = delta < 1
ms11 #Averaged the best 3 models

#Show coefficients of full model with CI
msfull <- cbind(summary(ms11)$coefmat.full, confint(ms11, full = TRUE))
msfull <- round(msfull, digits = 5)
msfull
write.csv(msfull,"D:\\Dropbox\\Bearbeitete Dokumente Uni Basel\\22Masterarbeit\\Schriftliche Arbeit\\MSfull.csv", row.names = FALSE)

#Show coefficients of subset model with CI
mssubset <- cbind(summary(ms11)$coefmat.subset, confint(ms11, full = FALSE))
mssubset <- round(mssubset, digits = 5)
mssubset
write.csv(mssubset,"D:\\Dropbox\\Bearbeitete Dokumente Uni Basel\\22Masterarbeit\\Schriftliche Arbeit\\MSsubset.csv", row.names = FALSE)

#Show the best model with best AICc
get.models(ms1, 1)#Object after dredge

#####################################################################################
################ 4.4 Perform averaged logit model with train data ###################
#####################################################################################
#model.avg(..., fit = TRUE) wrong because it fits models again (twice) when using it from model.selection command
#First step for using predict function:
library(tictoc)
tic()
#Create List of evaluated models ordered by AIC
bank2 <- lapply(dredge(bank1, rank = "AICc", evaluate = FALSE), eval)
toc()

tic()
#Average the models with a choosen set of models
SavedAveragedModels <- model.avg(bank2, subset = delta < 100)
toc()
SavedAveragedModels

#####################################################################################
################# 4.4 Perform average model.fit on train data #######################
#####################################################################################
bankaveragefittraindata.prob = predict(SavedAveragedModels, banktrainfull, type="response")

ROUNDbankaveragefittraindata.prob <- round(bankaveragefittraindata.prob, digits = 3)
bankaveragetrainpred <- ifelse(ROUNDbankaveragefittraindata.prob >= "0.5", 1, 0)
fbankaveragetrainpred <- factor(bankaveragetrainpred)
levels(fbankaveragetrainpred) = c('NoSubscription','Subscription')

#Use "fbanktrainreal" from logit example as reference (it is the same)

#Creating confusion matrix as a simple overview
ConfMataveragetrain <- confusionMatrix(data=fbankaveragetrainpred, reference = fbanktrainreal) #Reference = Real Data

#Display results 
ConfMataveragetrain

#Plot
fourfoldplot(ConfMataveragetrain$table, main = "Average Traindata")

#Plot Confusion Matrix with function "draw_confusion_matrix_bankaverage_train", execute all if not already done!
draw_confusion_matrix_bankaverage_train(ConfMataveragetrain)

######### Check if the same numbers as above
library(gmodels)
#Computes the crosstable calculations
CrossTable(fbankaveragetrainpred,fbanktrainreal)
#Check if same numbers
table(fbankaveragetrainpred)
table(fbanktrainreal)

#####################################################################################
################ 4.4 Perform average model.fit on test data #########################
#####################################################################################
bankaveragefittestdata.prob = predict(SavedAveragedModels, banktest, type="response")

ROUNDbankaveragefittestdata.prob <- round(bankaveragefittestdata.prob, digits = 3)
bankaveragetestpred <- ifelse(ROUNDbankaveragefittestdata.prob >= "0.5", 1, 0)
fbankaveragetestpred <- factor(bankaveragetestpred)
levels(fbankaveragetestpred) = c('NoSubscription','Subscription')

#Use "fbanktestreal" from logit example as reference (it is the same)

#Creating confusion matrix
ConfMataveragetest <- confusionMatrix(data=fbankaveragetestpred, reference = fbanktestreal)

#Display results 
ConfMataveragetest

#Plot
fourfoldplot(ConfMataveragetest$table, main = "Average Testdata")

#Plot Confusion Matrix with function "draw_confusion_matrix_bankaverage_test", execute all if not already done!
draw_confusion_matrix_bankaverage_test(ConfMataveragetest)

####### Check if the same numbers as above
library(gmodels)
#Computes the crosstable calculations
CrossTable(fbankaveragetestpred,fbanktestreal)
#Check numbers
table(fbankaveragetestpred)
table(fbanktestreal)

#####################################################################################
# 4.4 Comparison of AVERAGE Confusion Matrix - AVERAGE Insample VS AVERAGE Out-Of-Sample
#####################################################################################

#Exact
draw_confusion_matrix_bankaverage_train(ConfMataveragetrain)
draw_confusion_matrix_bankaverage_test(ConfMataveragetest)
#Plot
par(mfrow=c(1,2)) #2 graphs side by side
fourfoldplot(ConfMataveragetrain$table, main = "Average Traindata")
fourfoldplot(ConfMataveragetest$table, main = "Average Testdata")

#####################################################################################
###################### 4.5 Compare Coefficient tables ###############################
#####################################################################################
library(jtools)
library(kableExtra)
library(ggstance)
library(broom.mixed)
library(huxtable)
library(texreg)

#coefficient table (not possible in the same table)
screenreg(SavedAveragedModels, single.row = TRUE, custom.model.names = c("Full Averaged Model"))
#screenreg(banklogittraindata.fit, single.row = TRUE, custom.model.names = c("Logit Model"))
screenreg(banklogittraindata.fit, reorder.coef = c(1,25,23,24,21,3,4,5,6,7,8,9,10,11,12,13,22,14,15,27,28,29,19,16,17,18,2,26,20), single.row = TRUE, custom.model.names = c("Logit Model"))

#####################################################################################
############# 4.6 Confusion Matrix - Comparison of Testdata results #################
#####################################################################################
#Confusion Matrix - Comparison of Traindata results
#Exact (NOT SHOWN IN THE THESIS)
draw_confusion_matrix_bank_train(ConfMattrain)
draw_confusion_matrix_bankaverage_train(ConfMataveragetrain)
#Plot
par(mfrow=c(1,2)) #2 graphs side by side
fourfoldplot(ConfMattrain$table, main = "Logit Traindata")
fourfoldplot(ConfMataveragetrain$table, main = "Average Traindata")

#Confusion Matrix - Comparison of Testdata results
#Exact (NOT SHOWN IN THE THESIS)
draw_confusion_matrix_bank_test(ConfMattest)
draw_confusion_matrix_bankaverage_test(ConfMataveragetest)
#Plot
par(mfrow=c(1,2)) #2 graphs side by side
fourfoldplot(ConfMattest$table, main = "Logit Testdata")
fourfoldplot(ConfMataveragetest$table, main = "Average Testdata")

#####################################################################################
############################# 4.7 ROC of Performance ################################
#####################################################################################
library(ROCR)

############### Score train data which was predicted with logit.fit ##################
#Take the prediction "banklogitfittestdata.prob"
predlogittrain = prediction(banklogitfittraindata.prob, banktrainreal)

#Perform AUC
auclogittrain = performance(predlogittrain,"auc")@y.values[[1]][1]

#plot ROC curve of Logit Testdataprediction
perflogittrain <- performance(predlogittrain,"tpr","fpr")
plot(perflogittrain, colorize=TRUE, xlab="False Positive Rate", ylab="True Positive Rate", lwd = 1, cex.main=1,
     main= paste("Logistic Regression Traindata ROC Curve: AUC =", round(auclogittrain,5)))
abline(a=0, b = 1, col='darkorange1')


############### Score train data which was predicted with average.fit ##################
predAtrain = prediction(bankaveragefittraindata.prob, banktrainreal)

#Perform AUC
aucAtrain = performance(predAtrain,"auc")@y.values[[1]][1]

#plot ROC curve of averaged Traindataprediction
perfAtrain <- performance(predAtrain,"tpr","fpr")
plot(perfAtrain, col="blue",colorize=TRUE, xlab="False Positive Rate", ylab="True Positive Rate" ,lwd = 1 , cex.main=1,
     main= paste("Model Averaged Regression Traindata ROC Curve: AUC =", round(aucAtrain,5)))
abline(a=0, b = 1, col='darkorange1')


############### Score test data which was predicted with logit.fit ##################
#Take the prediction "banklogitfittestdata.prob"
predlogittest = prediction(banklogitfittestdata.prob, banktestreal)

#Perform AUC
auclogittest = performance(predlogittest,"auc")@y.values[[1]][1]

#plot ROC curve of Logit Testdataprediction
perflogittest <- performance(predlogittest,"tpr","fpr")
plot(perflogittest, colorize=TRUE, xlab="False Positive Rate", ylab="True Positive Rate", lwd = 1, cex.main=1,
     main= paste("Logistic Regression Testdata ROC Curve: AUC =", round(auclogittest,5)))
abline(a=0, b = 1, col='darkorange1')


############### Score test data which was predicted with average.fit ##################
predAtest = prediction(bankaveragefittestdata.prob, banktestreal)

#Perform AUC
aucAtest = performance(predAtest,"auc")@y.values[[1]][1]

#plot ROC curve of averaged Testdataprediction
perfAtest <- performance(predAtest,"tpr","fpr")
plot(perfAtest, col="blue", colorize=TRUE, xlab="False Positive Rate", ylab="True Positive Rate", lwd = 1, cex.main=1,
     main= paste("Model Averaged Regression Testdata ROC Curve: AUC =", round(aucAtest,5)))
abline(a=0, b = 1, col='darkorange1')

#If you want to see the different Thresholds in the plot, add this: text.adj=c(-0.2,1.2), print.cutoffs.at= seq(0,0.3,0.05)

#####################################################################################
############## 4.8 Distribution of Prediction Differences (Absolute) ################
#####################################################################################

#######Logit Bank Prediction
Logitfit.prob <- data.frame(banklogitfittestdata.prob)
bankreal <- data.frame(banktestreal)

LogitProb.diff <- cbind(bankreal,Logitfit.prob)
library(tidyverse)
LogitProb.diff <- LogitProb.diff %>% 
  mutate(Logitdiff = bankreal-Logitfit.prob)
LogitProb.diff <- LogitProb.diff %>% 
  mutate(absDiff = abs(Logitdiff))
#Averaged Deviation of all values
colMeans(LogitProb.diff$absDiff)

# Look at positive values which indicate a classification of 1 (Subscribe)
#Positiv 1 - Subscription
positiv <- subset(LogitProb.diff, banktestreal!=0)
colMeans(positiv$Logitdiff)

# Look at negative values which indicate a classification of 0 (Not Subscribe)
#Negativ 0 - No Subscription
negativ <- subset(LogitProb.diff, banktestreal!=1)
colMeans(negativ$Logitdiff)

#check if the results are correct => Yes
colMeans(LogitProb.diff$absDiff)
table(banktestreal)
#(2800*(-colMeans(negativ$Logitdiff))+365*colMeans(positiv$Logitdiff))/(2800+365)

#######Bank Average Prediction
Logitfit.probA <- data.frame(bankaveragefittestdata.prob)
bankreal <- data.frame(banktestreal)

LogitProb.diffA <- cbind(bankreal,Logitfit.probA)
library(tidyverse)
LogitProb.diffA <- LogitProb.diffA %>% 
  mutate(LogitdiffA = bankreal-Logitfit.probA)
LogitProb.diffA <- LogitProb.diffA %>% 
  mutate(absDiffA = abs(LogitdiffA))
#Averaged Deviation of all values
colMeans(LogitProb.diffA$absDiffA)

# Look at positive values which indicate a classification of 1 (Subscribe)
#Positiv 1 - Subscription
positivA <- subset(LogitProb.diffA, banktestreal!=0)
colMeans(positivA$LogitdiffA)

# Look at negative values which indicate a classification of 0 (Not Subscribe)
#Negativ 0 - No Subscription
negativA <- subset(LogitProb.diffA, banktestreal!=1)
colMeans(negativA$LogitdiffA)

#check if the results are correct => Yes
colMeans(LogitProb.diffA$absDiffA)
table(banktestreal)
#(2800*(-colMeans(negativA$LogitdiffA))+365*colMeans(positivA$LogitdiffA))/(2800+365)

### Copy all these values in a table in Master Thesis ###
Deviations <- as.data.frame(cbind(nrow(negativ), colMeans(negativ$Logitdiff), nrow(positiv), colMeans(positiv$Logitdiff), colMeans(LogitProb.diff$absDiff)))
DeviationsA <- as.data.frame(cbind(nrow(negativA), colMeans(negativA$LogitdiffA), nrow(positivA), colMeans(positivA$LogitdiffA), colMeans(LogitProb.diffA$absDiffA)))
DeviationsTable <- rbind(Deviations, DeviationsA)
colnames(DeviationsTable) <- c("N Negatives","Mean Negatives", "N Positives", "Mean Positives", "Arithmetic Mean")
rownames(DeviationsTable) <- c("Logit", "Average")
DeviationsTable
DeviationsTable <- round(DeviationsTable, digits = 7)
DeviationsTable
write.csv(DeviationsTable,"D:\\Dropbox\\Bearbeitete Dokumente Uni Basel\\22Masterarbeit\\Schriftliche Arbeit\\DeviationsTable.csv", row.names = FALSE)

############################## Plot the density of the calculated deviations ######
DeviationsPlot <- as.data.frame(LogitProb.diff$Logitdiff)
DeviationsPlotA <- as.data.frame(LogitProb.diffA$LogitdiffA)

d <- density(DeviationsPlot$banktestreal)
dA <- density(DeviationsPlotA$banktestreal)

par(mfrow=c(1,2)) #2 graphs side by side
h <-hist(DeviationsPlot$banktestreal, breaks=1000, xlim = c(-1,1), ylim = c(0,28), xlab = "Deviation", ylab = "Amount of same Deviations",
         main="Histogram Deviations of Predicted Outcome - Logit Test Data")
Meanh <- colMeans(LogitProb.diff$absDiff)
lines(c(Meanh,Meanh), c(0,10), col = "green", lwd = 3)#Mean
lines(d, lwd = 3, col="red")

#Legend
legend(0, 20, 
       c("Logit Deviation Density","Mean"), 
       lty=c(1,1), 
       lwd=c(4,4),
       col=c("red","green"))


############################## Plot the density of the calculated deviations AVERAGE
hA <-hist(DeviationsPlotA$banktestreal, breaks=1000, xlim = c(-1,1), ylim = c(0,28), xlab = "Deviation", ylab = "Amount of same Deviations",
          main="Histogram Deviations of Predicted Outcome - Average Test Data")
MeanhA <- colMeans(LogitProb.diffA$absDiffA)
lines(c(MeanhA,MeanhA), c(0,10), col = "green", lwd = 3)#Mean
lines(dA, lwd = 3, col="blue")

#Legend
legend(0, 20, 
       c("Average Deviation Density","Mean"), 
       lty=c(1,1), 
       lwd=c(4,4),
       col=c("blue","green"))


############### Density Comparison in same plot ##############################
par(mfrow=c(1,2)) #2 graphs
plot(d, col="red", type = "l", xlim = c(-1,1), ylim = c(0,6.5), main = "Density Comparison of Deviation - Logit", xlab="Deviation" , ylab="Density")
#Legend
legend(0.03,5, 
       c("Logit Deviation Density"), 
       lty=c(1), 
       lwd=c(4),
       col=c("red"))

plot(dA, col="blue", type = "l", xlim = c(-1,1), ylim = c(0,6.5), main = "Density Comparison of Deviation - Average", xlab="Deviation" , ylab="Density")
#Legend
legend(0.03,5, 
       c("Average Deviation Density"), 
       lty=c(1), 
       lwd=c(4),
       col=c("blue"))


#Raw plot of points
par(mfrow=c(2,1)) #2 graph over another
plot(x = DeviationsPlot$banktestreal, y = 1:length(DeviationsPlot$banktestreal), xlim = c(-1,1), xlab="Deviation", ylab="Observation")
plot(x = DeviationsPlotA$banktestreal, y = 1:length(DeviationsPlot$banktestreal), xlim = c(-1,1), xlab="Deviation", ylab="Observation")


#####################################################################################
########## 4.9 Loop with different sizes of Traindata and fixed Testdata ############
#####################################################################################

################################### LOGIT LOOP ######################################
#Create empty dataframe for loopresults
LoopResults <- data.frame(matrix(0, nrow = 5, ncol = 10))

#Split data into training and test
library(lattice)
library(ggplot2)
library(caret)
library(caTools)
for (k in 1:10){ #start of for loop with different k for different sample splits
  set.seed(125)
  sample = sample.split(bankdata$subscribed, SplitRatio = (.3)) #split into 30% train data and 70% testdata
  bankdatatrainfull = subset(bankdata, sample == TRUE)
  bankdatatest  = subset(bankdata, sample == FALSE)
  
  #set.seed(123)
  sample = sample.split(bankdatatrainfull$subscribed, SplitRatio = ((0.099999999)*k)) # split traindata again in parts*k
  bankdatatrainsubset = subset(bankdatatrainfull, sample == TRUE) #This leads to traindatasets of for example 3%, 6%, 9%, 12%.....30%
  bankdatatestunused  = subset(bankdatatrainfull, sample == FALSE)
  
  ##################################################
  #predict function - using the fitted function from training dataset on test dataset
  
  #logit regression with train data
  bankdatalogittraindata.fit = glm(subscribed ~ age + job + marital + education + default + balance + housing + loan + contact
                                   + campaign + previous + poutcome, control=glm.control(maxit=100), data=bankdatatrainsubset, family=binomial)
  summary(bankdatalogittraindata.fit)
  
  #Perform logit function from train data on test data
  bankdatalogitfittestdata.prob = predict(bankdatalogittraindata.fit, bankdatatest, type="response")
  
  #Round and prepare predicted results
  ROUNDbankdatalogitfittestdata.prob <- round(bankdatalogitfittestdata.prob, digits = 3)
  bankdatatestpred <- ifelse(ROUNDbankdatalogitfittestdata.prob >= "0.5", 1, 0)
  fbankdatatestpred <- factor(bankdatatestpred)
  levels(fbankdatatestpred) = c('NoSubscription','Subscription')
  
  #Prepare real results
  bankdatatestreal <- bankdatatest[,1]
  bankdatatestreal <- ifelse(bankdatatestreal >= "0.5", 1, 0)
  fbankdatatestreal <- factor(bankdatatestreal)
  levels(fbankdatatestreal) = c('NoSubscription','Subscription')
  
  
  #Creating confusion matrix
  ConfMat <- confusionMatrix(data=fbankdatatestpred, reference = fbankdatatestreal)
  
  #Display results 
  ConfMat
  
  #convert matrix into vector
  ConfMatResultsvector <- as.vector(t(ConfMat$table))
  ConfMatResults <- data.frame(ConfMatResultsvector)
  ConfMatResults #Just do the collected vectors in a data frame?
  ConfMatResults <- rbind(ConfMatResults, k*3)#add % of traindata
  
  LoopResults[1:5,k] <- data.frame(ConfMatResults) #Do not change, even if you split into more than K=10 datasets!
  #End of for-loop
}

#EXTRACT AND PLOT
ConfMatResultsComplete <- t(LoopResults)


colnames(ConfMatResultsComplete) <- c("A_true_positive", "B_false_positive", "C_false_negative", "D_true_negative", "Size_Training_data_percent")
ConfMatResultsComplete

ConfMatResultsComplete <- data.frame(ConfMatResultsComplete)


#USE ACCURACY, SENSITIVITY AND SPECIFITY INSTEAD
library(tidyverse)
ConfMatResultsComplete <- ConfMatResultsComplete %>% 
  mutate(sample_size = A_true_positive + B_false_positive + C_false_negative + D_true_negative)

ConfMatResultsComplete <- ConfMatResultsComplete %>% 
  mutate(Accuracy = (A_true_positive + D_true_negative)/sample_size)

ConfMatResultsComplete <- ConfMatResultsComplete %>% 
  mutate(sensitivity_recall_TPR = (A_true_positive/(A_true_positive + C_false_negative)))

ConfMatResultsComplete <- ConfMatResultsComplete %>% 
  mutate(specificity_selectivity_TNR = (D_true_negative/(D_true_negative + B_false_positive)))

ConfMatResultsComplete <- ConfMatResultsComplete %>% 
  mutate(FPR = (1-specificity_selectivity_TNR))



library(ggplot2)
library(graphics)


########################### Relative Plot #########################################
#Plot Accuracy
par(mfrow=c(2,1)) #1 graph on top and the other below
plot(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$Accuracy, main="BankData Logit Accuracy",
     xlab="Size of training sample in %", ylab="Accuracy",type="o", col="red", pch="o", lty=1, ylim=c(0,1))

#Plot Sensitivity_recall_TPR
points(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$sensitivity_recall_TPR, col="blue", pch="+")
lines(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$sensitivity_recall_TPR, col="blue",lty=2)

#Plot Specificity_TNR
points(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$specificity_selectivity_TNR, col="dark blue",pch="*")
lines(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$specificity_selectivity_TNR, col="dark blue", lty=3)

#legend
legend(22,0.65,legend=c("Accuracy","TPR_Sensitivity_Recall","TNR_Specificity"), col=c("red","blue","dark blue"),pch=c("o","+","*"),lty=c(1,2,3), ncol=1)

#############Plot all absolute values of confusion matrix##################
#Plot A_true_positive
plot(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$A_true_positive, main="BankData Logit Absolute",
     xlab="Size of training sample in %", ylab="Number of absolute Values",type="o", col="red", pch="o", lty=1, ylim=c(0, 3400))

#Plot C_false_negative
points(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$C_false_negative, col="blue", pch="*")
lines(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$C_false_negative, col="blue",lty=2)

#Plot D_true_negative
points(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$D_true_negative, col="dark blue",pch="o")
lines(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$D_true_negative, col="dark blue", lty=3)

#Plot B_false_positive
points(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$B_false_positive, col="dark green",pch="+")
lines(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$B_false_positive, col="dark green", lty=3)

#legend
legend(22,2500,legend=c("A_true_positive", "D_true_negative", "C_false_negative", "B_false_positive"), col=c("red","dark blue","blue","dark green"),pch=c("o","o","*","+"),lty=c(1,2,3), ncol=1)


################################### AVERAGE LOOP ######################################

#Create empty dataframe for Averageloopresults
LoopaverageResults <- data.frame(matrix(0, nrow = 5, ncol = 10))

library(lattice)
library(ggplot2)
library(caret)
library(caTools)
library(tictoc)
for (k in 1:10){ #start of for loop with different k for different sample splits
  tic()
  set.seed(125)
  sample = sample.split(bankdata$subscribed, SplitRatio = (.3)) #split into 30% train data and 70% testdata
  bankdataaveragetrainfull = subset(bankdata, sample == TRUE)
  bankdataaveragetest  = subset(bankdata, sample == FALSE)
  
  sample = sample.split(bankdataaveragetrainfull$subscribed, SplitRatio = ((0.099999999)*k)) # split traindata again in 10% ####1% = (10/3)*k
  bankdataaveragetrainsubset = subset(bankdataaveragetrainfull, sample == TRUE) #This leads to traindatasets of 3%, 6%, 9%, 12%.....30%
  bankdataaveragetestunused  = subset(bankdataaveragetrainfull, sample == FALSE)
  
  
  ##################################################
  #predict function - using the fitted function from training dataset on test dataset
  
  library(MuMIn)

  fm1 <- glm(subscribed ~ age + job + marital + education + default + balance + housing + loan + contact
             + campaign + previous + poutcome, data=bankdataaveragetrainsubset, family=binomial, control=glm.control(maxit=100), na.action = na.fail)

  #First step for using predict function
  ####tic()
  ms11 <- lapply(dredge(fm1, evaluate = FALSE), eval)
  ###toc()
  
  ##tic()
  SavedAveragedModels <- model.avg(ms11, subset = delta < 100)
  #toc()
  SavedAveragedModels
  
  #Perform on test data
  bankdataaveragelogitfittestdata.prob = predict(SavedAveragedModels, bankdataaveragetest, type="response")
  
  #Round and prepare predicted results
  ROUNDbankdataaveragelogitfittestdata.prob <- round(bankdataaveragelogitfittestdata.prob, digits = 3)
  bankdataaveragetestpred <- ifelse(ROUNDbankdataaveragelogitfittestdata.prob >= "0.5", 1, 0)
  fbankdataaveragetestpred <- factor(bankdataaveragetestpred)
  levels(fbankdataaveragetestpred) = c('NoSubscription','Subscription')
  
  #Prepare real results
  bankdataaveragetestreal <- bankdataaveragetest[,1]
  bankdataaveragetestreal <- ifelse(bankdataaveragetestreal >= "0.5", 1, 0)
  fbankdataaveragetestreal <- factor(bankdataaveragetestreal)
  levels(fbankdataaveragetestreal) = c('NoSubscription','Subscription')
  
  #Creating confusion matrix
  ConfMataverage <- confusionMatrix(data=fbankdataaveragetestpred, reference = fbankdataaveragetestreal)
  
  #Display results 
  ConfMataverage
  
  #convert matrix into vector
  ConfMataverageResultsvector <- as.vector(t(ConfMataverage$table))
  ConfMataverageResults <- data.frame(ConfMataverageResultsvector)
  ConfMataverageResults #Just do the collected vectors in a data frame
  ConfMataverageResults <- rbind(ConfMataverageResults, k*3) #k=10 for 10%, 5 for 5%
  ConfMataverageResults
  
  LoopaverageResults[1:5,k] <- data.frame(ConfMataverageResults) #No need for a empty dataframe because we can change it from a matrix
  #End of for-loop
  toc()
}

#EXTRACT AND PLOT
#ConfMatResults <- data.frame(ConfMatResults)
ConfMataverageResultsComplete <- t(LoopaverageResults)

colnames(ConfMataverageResultsComplete) <- c("A_true_positive", "B_false_positive", "C_false_negative", "D_true_negative", "Size_Training_data_percent")
ConfMataverageResultsComplete

ConfMataverageResultsComplete <- data.frame(ConfMataverageResultsComplete)

#USE ACCURACY, SENSITIVITY AND SPECIFITY INSTEAD
library(tidyverse)
ConfMataverageResultsComplete <- ConfMataverageResultsComplete %>% 
  mutate(sample_size = A_true_positive + B_false_positive + C_false_negative + D_true_negative)

ConfMataverageResultsComplete <- ConfMataverageResultsComplete %>% 
  mutate(Accuracy = (A_true_positive + D_true_negative)/sample_size)

ConfMataverageResultsComplete <- ConfMataverageResultsComplete %>% 
  mutate(sensitivity_recall_TPR = (A_true_positive/(A_true_positive + C_false_negative)))

ConfMataverageResultsComplete <- ConfMataverageResultsComplete %>% 
  mutate(specificity_selectivity_TNR = (D_true_negative/(D_true_negative + B_false_positive)))



library(ggplot2)
library(graphics)

########################### Relative Plot #########################################
#Plot Accuracy
par(mfrow=c(1,1)) #1 graph on top and the other below
plot(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$Accuracy, main="BankData Average Logit Accuracy",
     xlab="Size of training sample in %", ylab="Accuracy", type="o", col="red", pch="o", lty=1, ylim=c(0,1) )

#Plot Sensitivity_recall_TPR
points(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$sensitivity_recall_TPR, col="blue", pch="+")
lines(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$sensitivity_recall_TPR, col="blue",lty=2)

#Plot Specificity_TNR
points(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$specificity_selectivity_TNR, col="dark blue",pch="*")
lines(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$specificity_selectivity_TNR, col="dark blue", lty=3)

#legend
legend(22,0.65,legend=c("Accuracy","TPR_Sensitivity_Recall","TNR_Specificity"), col=c("red","blue","dark blue"),pch=c("o","+","*"),lty=c(1,2,3), ncol=1)

#############Plot all absolute values of confusion matrix##################
#Plot A_true_positive
plot(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$A_true_positive, main="BankData Average Logit Absolute",
     xlab="Size of training sample in %", ylab="Number of absolute Values",type="o", col="red", pch="o", lty=1, ylim=c(0, 1200))

#Plot C_false_negative
points(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$C_false_negative, col="blue", pch="*")
lines(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$C_false_negative, col="blue",lty=2)

#Plot D_true_negative
points(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$D_true_negative, col="dark blue",pch="o")
lines(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$D_true_negative, col="dark blue", lty=3)

#Plot B_false_positive
points(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$B_false_positive, col="dark green",pch="+")
lines(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$B_false_positive, col="dark green", lty=3)

#legend
legend(22,2500,legend=c("A_true_positive", "D_true_negative", "C_false_negative", "B_false_positive"), col=c("red","dark blue","blue","dark green"),pch=c("o","o","*","+"),lty=c(1,2,3), ncol=1)


#####################################################################################
##################### 4.9 Comparison of Learning Curve ##############################
#####################################################################################

################## Logit Learning Curve #####################
###################### LOGIT  Relative Plot #########################################
#Plot Accuracy
par(mfrow=c(2,2)) #1 graph on top and the other below
plot(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$Accuracy, main="BankData Logit Accuracy", ylab="Accuracy", xlab="Size of training sample in %",type="o", col="red", pch="o", lty=1, ylim=c(0,1))

#Plot Sensitivity_Recall_TPR
points(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$sensitivity_recall_TPR, col="blue", pch="+")
lines(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$sensitivity_recall_TPR, col="blue",lty=2)

#Plot Specificity_TNR
points(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$specificity_selectivity_TNR, col="dark blue",pch="*")
lines(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$specificity_selectivity_TNR, col="dark blue", lty=3)

#legend
legend(20,0.65,legend=c("Accuracy","TPR_Sensitivity_Recall","TNR_Specificity"), col=c("red","blue","dark blue"),pch=c("o","+","*"),lty=c(1,2,3), ncol=1)

############## AVERAGE Relative Plot #########################################
#Plot Accuracy
#par(mfrow=c(2,1)) #1 graph on top and the other below
plot(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$Accuracy, main="BankData Average Logit Accuracy",
     xlab="Size of training sample in %", ylab="Accuracy", type="o", col="red", pch="o", lty=1, ylim=c(0,1) )

#Plot Sensitivity_Recall_TPR
points(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$sensitivity_recall_TPR, col="blue", pch="+")
lines(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$sensitivity_recall_TPR, col="blue",lty=2)

#Plot Specificity_TNR
points(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$specificity_selectivity_TNR, col="dark blue",pch="*")
lines(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$specificity_selectivity_TNR, col="dark blue", lty=3)

#legend
legend(20,0.65,legend=c("Accuracy","TPR_Sensitivity_Recall","TNR_Specificity"), col=c("red","blue","dark blue"),pch=c("o","+","*"),lty=c(1,2,3), ncol=1)

################ LOGIT Plot all absolute values of confusion matrix##################
#Plot A_true_positive
plot(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$A_true_positive, main="BankData Logit Absolute",
     xlab="Size of training sample in %", ylab="Number of absolute Values",type="o", col="red", pch="o", lty=1, ylim=c(0, 3000))

#Plot C_false_negative
points(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$C_false_negative, col="blue", pch="*")
lines(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$C_false_negative, col="blue",lty=2)

#Plot D_true_negative
points(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$D_true_negative, col="dark blue",pch="o")
lines(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$D_true_negative, col="dark blue", lty=3)

#Plot B_false_positive
points(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$B_false_positive, col="dark green",pch="+")
lines(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$B_false_positive, col="dark green", lty=3)

#legend
legend(20,2500,legend=c("A_true_positive", "D_true_negative", "C_false_negative", "B_false_positive"), col=c("red","dark blue","blue","dark green"),pch=c("o","o","*","+"),lty=c(1,2,3), ncol=1)

########## AVERAGE Plot all absolute values of confusion matrix ##############
#Plot A_true_positive
plot(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$A_true_positive, main="BankData Average Logit Absolute",
     xlab="Size of training sample in %", ylab="Number of absolute Values",type="o", col="red", pch="o", lty=1, ylim=c(0, 3000))

#Plot C_false_negative
points(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$C_false_negative, col="blue", pch="*")
lines(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$C_false_negative, col="blue",lty=2)

#Plot D_true_negative
points(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$D_true_negative, col="dark blue",pch="o")
lines(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$D_true_negative, col="dark blue", lty=3)

#Plot B_false_positive
points(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$B_false_positive, col="dark green",pch="+")
lines(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$B_false_positive, col="dark green", lty=3)

#legend
legend(20,2500,legend=c("A_true_positive", "D_true_negative", "C_false_negative", "B_false_positive"), col=c("red","dark blue","blue","dark green"),pch=c("o","o","*","+"),lty=c(1,2,3), ncol=1)


#####################################################################################
################# 5. Further Examples with other Datasets ###########################
#####################################################################################

######################### Example 5.1: GermanCredit ##################################
rm(list = ls())
WorkDir <- ("D:/Dropbox/Bearbeitete Dokumente Uni Basel/22Masterarbeit/Code")
setwd(WorkDir)
options(mc.cores = parallel::detectCores())
###############################################################################
########################## Load and inspect Dataset ###########################
###############################################################################
library(caret)
data(GermanCredit)
library(tidyverse)
creditquality = GermanCredit %>% rename(creditquality = Class)
creditquality$creditquality <- ifelse(creditquality$creditquality == "Good", 1, 0)
creditquality = creditquality[,c(10,1:9,11:50)]
creditquality = na.omit(creditquality)
table(creditquality$creditquality)
#####################################################################################
############# Loop with different sizes of Traindata and fixed Testdata #############
#####################################################################################
#Duration of this chapter is around 100minutes when running the code on my PC
################################### LOGIT LOOP ######################################
#Create empty dataframe for loopresults
LoopResults <- data.frame(matrix(0, nrow = 5, ncol = 90))
library(lattice)
library(ggplot2)
library(caret)
library(caTools)
for (k in 1:90){ #start of for loop with different k for different sample splits
  set.seed(900)
  sample = sample.split(creditquality$creditquality, SplitRatio = (.9)) #split into 90% train data and 10% testdata
  creditqualitytrainfull = subset(creditquality, sample == TRUE)
  creditqualitytest  = subset(creditquality, sample == FALSE)
  
  sample = sample.split(creditqualitytrainfull$creditquality, SplitRatio = ((0.099999999/(9))*k)) # split traindata again
  creditqualitytrainsubset = subset(creditqualitytrainfull, sample == TRUE) 
  creditqualitytestunused  = subset(creditqualitytrainfull, sample == FALSE)
  
  ##################################################
  #predict function - using the fitted function from training dataset on test dataset
  #logit regression with train data
  creditqualitylogittraindata.fit = glm(creditquality ~
                                          Duration + Amount + Age + ForeignWorker + EmploymentDuration.Unemployed +
                                          CreditHistory.Critical + Personal.Male.Single +
                                          Purpose.UsedCar + OtherDebtorsGuarantors.None + CheckingAccountStatus.none + SavingsAccountBonds.gt.1000 +
                                          Property.RealEstate, control=glm.control(maxit=100), data=creditqualitytrainsubset, family=binomial)
  summary(creditqualitylogittraindata.fit)
  
  #Perform logit function from train data on test data
  creditqualitylogitfittestdata.prob = predict(creditqualitylogittraindata.fit, creditqualitytest, type="response")
  
  #Round and prepare predicted results
  ROUNDcreditqualitylogitfittestdata.prob <- round(creditqualitylogitfittestdata.prob, digits = 3)
  creditqualitytestpred <- ifelse(ROUNDcreditqualitylogitfittestdata.prob >= "0.5", 1, 0)
  fcreditqualitytestpred <- factor(creditqualitytestpred)
  levels(fcreditqualitytestpred) = c('Bad','Good')
  
  #Prepare real results
  creditqualitytestreal <- creditqualitytest[,1]
  creditqualitytestreal <- ifelse(creditqualitytestreal >= "0.5", 1, 0)
  fcreditqualitytestreal <- factor(creditqualitytestreal)
  levels(fcreditqualitytestreal) = c('Bad','Good')
  
  
  #Creating confusion matrix
  ConfMat <- confusionMatrix(data=fcreditqualitytestpred, reference = fcreditqualitytestreal)
  
  #Display results 
  ConfMat
  
  #convert matrix into vector
  ConfMatResultsvector <- as.vector(t(ConfMat$table))
  ConfMatResults <- data.frame(ConfMatResultsvector)
  ConfMatResults #Just do the collected vectors in a data frame?
  ConfMatResults <- rbind(ConfMatResults, k*1)
  
  LoopResults[1:5,k] <- data.frame(ConfMatResults) #Do not change, even if you split into more than K=10 datasets!
  #End of for-loop
}

#EXTRACT AND PLOT
ConfMatResultsComplete <- t(LoopResults)


colnames(ConfMatResultsComplete) <- c("A_true_positive", "B_false_positive", "C_false_negative", "D_true_negative", "Size_Training_data_percent")
ConfMatResultsComplete

ConfMatResultsComplete <- data.frame(ConfMatResultsComplete)


#USE ACCURACY, SENSITIVITY AND SPECIFITY INSTEAD
library(tidyverse)
ConfMatResultsComplete <- ConfMatResultsComplete %>% 
  mutate(sample_size = A_true_positive + B_false_positive + C_false_negative + D_true_negative)

ConfMatResultsComplete <- ConfMatResultsComplete %>% 
  mutate(Accuracy = (A_true_positive + D_true_negative)/sample_size)

ConfMatResultsComplete <- ConfMatResultsComplete %>% 
  mutate(sensitivity_recall_TPR = (A_true_positive/(A_true_positive + C_false_negative)))

ConfMatResultsComplete <- ConfMatResultsComplete %>% 
  mutate(specificity_selectivity_TNR = (D_true_negative/(D_true_negative + B_false_positive)))

ConfMatResultsComplete <- ConfMatResultsComplete %>% 
  mutate(FPR = (1-specificity_selectivity_TNR))

### Plots can be fount at the end of the second loop

################################### AVERAGE LOOP ######################################
#Create empty dataframe for Averageloopresults
LoopaverageResults <- data.frame(matrix(0, nrow = 5, ncol = 90))
library(lattice)
library(ggplot2)
library(caret)
library(caTools)
library(tictoc)
for (k in 1:90){ #start of for loop with different k for different sample splits
  tic()
  set.seed(900)
  sample = sample.split(creditquality$creditquality, SplitRatio = (.9)) #split into 90% train data and 10% testdata
  creditqualityaveragetrainfull = subset(creditquality, sample == TRUE)
  creditqualityaveragetest  = subset(creditquality, sample == FALSE)
  
  sample = sample.split(creditqualityaveragetrainfull$creditquality, SplitRatio = ((0.099999999/(9))*k)) # split traindata again
  creditqualityaveragetrainsubset = subset(creditqualityaveragetrainfull, sample == TRUE) #This leads to traindatasets of 1%, 2%, 3%, 4%.....90%
  creditqualityaveragetestunused  = subset(creditqualityaveragetrainfull, sample == FALSE)
  
  
  ##################################################
  #predict function - using the fitted function from training dataset on test dataset
  
  library(MuMIn)
  
  fm1 <- glm(creditquality ~
               Duration + Amount + Age + ForeignWorker + EmploymentDuration.Unemployed +
               CreditHistory.Critical + Personal.Male.Single +
               Purpose.UsedCar + OtherDebtorsGuarantors.None + CheckingAccountStatus.none + SavingsAccountBonds.gt.1000 +
               Property.RealEstate, data=creditqualityaveragetrainsubset, family=binomial, control=glm.control(maxit=100), na.action = na.fail)
  
  #First step for using predict function
  ####tic()
  ms11 <- lapply(dredge(fm1, evaluate = FALSE), eval)
  ###toc()
  
  ##tic()
  SavedAveragedModels <- model.avg(ms11, subset = delta < 100)
  #toc()
  SavedAveragedModels
  
  #Perform on test data
  creditqualityaveragelogitfittestdata.prob = predict(SavedAveragedModels, creditqualityaveragetest, type="response")
  
  #Round and prepare predicted results
  ROUNDcreditqualityaveragelogitfittestdata.prob <- round(creditqualityaveragelogitfittestdata.prob, digits = 3)
  creditqualityaveragetestpred <- ifelse(ROUNDcreditqualityaveragelogitfittestdata.prob >= "0.5", 1, 0)
  fcreditqualityaveragetestpred <- factor(creditqualityaveragetestpred)
  levels(fcreditqualityaveragetestpred) = c('Bad','Good')
  
  #Prepare real results
  creditqualityaveragetestreal <- creditqualityaveragetest[,1]
  creditqualityaveragetestreal <- ifelse(creditqualityaveragetestreal >= "0.5", 1, 0)
  fcreditqualityaveragetestreal <- factor(creditqualityaveragetestreal)
  levels(fcreditqualityaveragetestreal) = c('Bad','Good')
  
  #Creating confusion matrix
  ConfMataverage <- confusionMatrix(data=fcreditqualityaveragetestpred, reference = fcreditqualityaveragetestreal)
  
  #Display results 
  ConfMataverage
  
  #convert matrix into vector
  ConfMataverageResultsvector <- as.vector(t(ConfMataverage$table))
  ConfMataverageResults <- data.frame(ConfMataverageResultsvector)
  ConfMataverageResults #Just do the collected vectors in a data frame
  ConfMataverageResults <- rbind(ConfMataverageResults, k*1) #10 for 10%, 5 for 5%
  ConfMataverageResults
  
  LoopaverageResults[1:5,k] <- data.frame(ConfMataverageResults) #No need for a empty dataframe because we can change it from a matrix
  #End of for-loop
  toc()
}

#EXTRACT AND PLOT
ConfMataverageResultsComplete <- t(LoopaverageResults)

colnames(ConfMataverageResultsComplete) <- c("A_true_positive", "B_false_positive", "C_false_negative", "D_true_negative", "Size_Training_data_percent")
ConfMataverageResultsComplete

ConfMataverageResultsComplete <- data.frame(ConfMataverageResultsComplete)

#USE ACCURACY, SENSITIVITY AND SPECIFITY INSTEAD
library(tidyverse)
ConfMataverageResultsComplete <- ConfMataverageResultsComplete %>% 
  mutate(sample_size = A_true_positive + B_false_positive + C_false_negative + D_true_negative)

ConfMataverageResultsComplete <- ConfMataverageResultsComplete %>% 
  mutate(Accuracy = (A_true_positive + D_true_negative)/sample_size)

ConfMataverageResultsComplete <- ConfMataverageResultsComplete %>% 
  mutate(sensitivity_recall_TPR = (A_true_positive/(A_true_positive + C_false_negative)))

ConfMataverageResultsComplete <- ConfMataverageResultsComplete %>% 
  mutate(specificity_selectivity_TNR = (D_true_negative/(D_true_negative + B_false_positive)))

#####################################################################################
################## Comparison of Learning Curve in 1 Plot ###########################
#####################################################################################
library(ggplot2)
library(graphics)
################### Logit Relative Plot #########################################
#Plot Accuracy
par(mfrow=c(2,2)) #1 graph on top and the other below
plot(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$Accuracy, main="CreditQuality Logit Accuracy",
     xlab="Size of training sample in %", ylab="Accuracy",type="o", col="red", pch="o", lty=1, ylim=c(0.22,1))

#Plot Sensitivity_Recall_TPR
points(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$sensitivity_recall_TPR, col="blue", pch="+")
lines(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$sensitivity_recall_TPR, col="blue",lty=2)

#Plot Specificity_TNR
points(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$specificity_selectivity_TNR, col="dark blue",pch="*")
lines(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$specificity_selectivity_TNR, col="dark blue", lty=3)

#legend
legend(40,0.74,legend=c("Accuracy","TPR_Sensitivity_Recall","TNR_Specificity"), col=c("red","blue","dark blue"),pch=c("o","+","*"),lty=c(1,2,3), ncol=1)

################## AVERAGE - Relative Plot #########################################
#Plot Accuracy
plot(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$Accuracy, main="CreditQuality Average Logit Accuracy",
     xlab="Size of training sample in %", ylab="Accuracy", type="o", col="red", pch="o", lty=1, ylim=c(0.22,1) )

#Plot Sensitivity_Recall_TPR
points(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$sensitivity_recall_TPR, col="blue", pch="+")
lines(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$sensitivity_recall_TPR, col="blue",lty=2)

#Plot Specificity_TNR
points(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$specificity_selectivity_TNR, col="dark blue",pch="*")
lines(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$specificity_selectivity_TNR, col="dark blue", lty=3)

#legend
legend(40,0.74,legend=c("Accuracy","TPR_Sensitivity_Recall","TNR_Specificity"), col=c("red","blue","dark blue"),pch=c("o","+","*"),lty=c(1,2,3), ncol=1)


############# Logit - Plot all absolute values of confusion matrix############
#Plot A_true_positive
plot(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$A_true_positive, main="CreditQuality Logit Absolute",
     xlab="Size of training sample in %", ylab="Number of absolute Values",type="o", col="red", pch="o", lty=1, ylim=c(0, 69))

#Plot C_false_negative
points(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$C_false_negative, col="blue", pch="*")
lines(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$C_false_negative, col="blue",lty=2)

#Plot D_true_negative
points(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$D_true_negative, col="dark blue",pch="o")
lines(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$D_true_negative, col="dark blue", lty=3)

#Plot B_false_positive
points(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$B_false_positive, col="dark green",pch="+")
lines(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$B_false_positive, col="dark green", lty=3)

#legend
legend(40,50,legend=c("A_true_positive", "D_true_negative", "C_false_negative", "B_false_positive"), col=c("red","dark blue","blue","dark green"),pch=c("o","o","*","+"),lty=c(1,2,3), ncol=1)


########## AVERAGE - Plot all absolute values of confusion matrix#############
#Plot A_true_positive
plot(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$A_true_positive, main="CreditQuality Average Logit Absolute",
     xlab="Size of training sample in %", ylab="Number of absolute Values",type="o", col="red", pch="o", lty=1, ylim=c(0, 69))

#Plot C_false_negative
points(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$C_false_negative, col="blue", pch="*")
lines(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$C_false_negative, col="blue",lty=2)

#Plot D_true_negative
points(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$D_true_negative, col="dark blue",pch="o")
lines(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$D_true_negative, col="dark blue", lty=3)

#Plot B_false_positive
points(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$B_false_positive, col="dark green",pch="+")
lines(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$B_false_positive, col="dark green", lty=3)

#legend
legend(40,50,legend=c("A_true_positive", "D_true_negative", "C_false_negative", "B_false_positive"), col=c("red","dark blue","blue","dark green"),pch=c("o","o","*","+"),lty=c(1,2,3), ncol=1)

###############################################################################
# Split dataset - 90% train data (Insample) und 10% test data (Out-of-sample) #
###############################################################################
library(caTools)
set.seed(900)
sample = sample.split(creditquality$creditquality, SplitRatio = (.9))
creditqualitytrainfull = subset(creditquality, sample == TRUE)
creditqualitytest  = subset(creditquality, sample == FALSE)

#####################################################################################
############################### Create Logit model ##################################
#####################################################################################
#logit regression with train data
creditqualitylogittraindata.fit = glm(creditquality ~
                                        Duration + Amount + Age + ForeignWorker + EmploymentDuration.Unemployed +
                                        CreditHistory.Critical + Personal.Male.Single +
                                        Purpose.UsedCar + OtherDebtorsGuarantors.None + CheckingAccountStatus.none + SavingsAccountBonds.gt.1000 +
                                        Property.RealEstate, control=glm.control(maxit=100), data=creditqualitytrainfull, family=binomial)
#Output and Coefficients
summary(creditqualitylogittraindata.fit)
library(jtools)
library(kableExtra)
library(ggstance)
library(broom.mixed)
library(huxtable)
library(texreg)
#Nice summary
summ(creditqualitylogittraindata.fit) #clear overview

#Additional ways to show coefficients
#export_summs(creditqualitylogittraindata.fit, scale = TRUE) #for combining models in 1 table
plot_summs(creditqualitylogittraindata.fit, scale = TRUE) #coefficient comparison (just best AIC and full model, average model not compatible!)
#Additional ways to show proper output
#screenreg(creditqualitylogittraindata.fit) #used for comparison later
#####################################################################################
############### Perform logit.fit from train data on train data #####################
#####################################################################################
creditqualitylogitfittraindata.prob = predict(creditqualitylogittraindata.fit, creditqualitytrainfull, type="response")

#Round and shape the predicted probabilities
ROUNDcreditqualitylogitfittraindata.prob <- round(creditqualitylogitfittraindata.prob, digits = 3)
creditqualitytrainpred <- ifelse(ROUNDcreditqualitylogitfittraindata.prob >= "0.5", 1, 0)
fcreditqualitytrainpred <- factor(creditqualitytrainpred)
levels(fcreditqualitytrainpred) = c('Bad','Good')

#Prepare the real outcomes
creditqualitytrainreal <- creditqualitytrainfull[,1]
fcreditqualitytrainreal <- factor(creditqualitytrainreal)
levels(fcreditqualitytrainreal) = c('Bad','Good')

#Creating confusion matrix - how performs the model on the train data - Insample
library(lattice)
library(ggplot2)
library(caret)
ConfMattrain <- confusionMatrix(data=fcreditqualitytrainpred, reference = fcreditqualitytrainreal)#Reference = Real data

#Display results in simple form
ConfMattrain

#####################################################################################
####################### Perform logit model on test data ############################
#####################################################################################
#Perform logit.fit on train data on test data
creditqualitylogitfittestdata.prob = predict(creditqualitylogittraindata.fit, creditqualitytest, type="response")

#Round and prepare predicted probabilities
ROUNDcreditqualitylogitfittestdata.prob <- round(creditqualitylogitfittestdata.prob, digits = 3)
creditqualitytestpred <- ifelse(ROUNDcreditqualitylogitfittestdata.prob >= "0.5", 1, 0)
fcreditqualitytestpred <- factor(creditqualitytestpred)
levels(fcreditqualitytestpred) = c('Bad','Good')

creditqualitytestreal <- creditqualitytest[,1]
fcreditqualitytestreal <- factor(creditqualitytestreal)
levels(fcreditqualitytestreal) = c('Bad','Good')

#Creating confusion matrix - how performs the model on the test data - Out-of-sample
ConfMattest <- confusionMatrix(data=fcreditqualitytestpred, reference = fcreditqualitytestreal)

#Display results 
ConfMattest

#####################################################################################
#Compare Confusion Matrix of Traindata and Testdata - Logit Insample VS Out-Of-Sample
#####################################################################################
#Plot Confusion Matrix with function "draw_confusion_matrix" for CreditQuality - execute all if not already done!
#Exact
draw_confusion_matrix_creditquality_train(ConfMattrain)
draw_confusion_matrix_creditquality_test(ConfMattest)
#Plot
par(mfrow=c(1,2)) #2 graphs side by side
fourfoldplot(ConfMattrain$table, main = "Logit Traindata")
fourfoldplot(ConfMattest$table, main = "Logit Testdata")


#####################################################################################
######################### Perform Average Method ####################################
#####################################################################################
library(MuMIn)
creditquality1 <- glm(creditquality ~
                        Duration + Amount + Age + ForeignWorker + EmploymentDuration.Unemployed +
                        CreditHistory.Critical + Personal.Male.Single +
                        Purpose.UsedCar + OtherDebtorsGuarantors.None + CheckingAccountStatus.none + SavingsAccountBonds.gt.1000 +
                        Property.RealEstate, data=creditqualitytrainfull, family=binomial, control=glm.control(maxit=100), na.action = na.fail)

#####################################################################################
################## MODEL INSPECTION WITH DIFFERENT COMMANDS #########################
#####################################################################################
#Normal fitting, but cant use it for predict function in this form
#model selection: you can combine models manually(model.sel) or automatically over "dredge"
(mscreditquality1 <- dredge(creditquality1, rank = "AICc", extra = c("R^2","adjR^2", F = function(x) #Create Model Selection table
  summary(x)$fstatistik[[1]])))
mscreditquality1 #Show model selection table
mscreditquality1[1:4] #show model selection table but new weights!

#Plot a graph of all models and their weights
par(mfrow=c(1,1)) #Plot just 1 graph
par(oma = c(0,1,2,2)) #increase distance to window borders to prevent cutted plot
plot(mscreditquality1) #Make plot window much bigger to plot without cutting

#After creating a model selection table, you can use "model.avg" to choose a subset of models
mscreditquality11 <- model.avg(mscreditquality1, subset = delta < 1) #Averaging the models with subset = delta < 1
mscreditquality11

#Show coefficients of full model with CI
msfull <- cbind(summary(mscreditquality11)$coefmat.full, confint(mscreditquality11, full = TRUE))
round(msfull, digits = 8)

#Show coefficients of full model with CI
mssubset <- cbind(summary(mscreditquality11)$coefmat.subset, confint(mscreditquality11, full = FALSE))
round(mssubset, digits = 8)

#Show the best model with best AICc
get.models(mscreditquality1, 1)#Object after dredge

#Subset best 4 models
library(texreg)
mscreditquality1Subset <- mscreditquality1[1:4, ]
screenreg(mscreditquality1Subset, custom.model.names = c("Best Model", "2. Best Model", "3. Best Model", "4. Best Model"))#show best 4 models


#####################################################################################
################## Perform averaged logit model on train data to check ##############
#####################################################################################
#First step for using predict function:
library(tictoc)
tic()
#Create List of evaluated models ordered by AIC
bank11 <- lapply(dredge(creditquality1, rank = "AICc", evaluate = FALSE), eval)
toc()

tic()
#Average the models with a choosen set of models
SavedAveragedModels <- model.avg(bank11, subset = delta < 100)
toc()
SavedAveragedModels

#####################################################################################
##################### Perform average model.fit on train data #######################
#####################################################################################
creditqualityaveragefittraindata.prob = predict(SavedAveragedModels, creditqualitytrainfull, type="response")

ROUNDcreditqualityaveragefittraindata.prob <- round(creditqualityaveragefittraindata.prob, digits = 3)
creditqualityaveragetrainpred <- ifelse(ROUNDcreditqualityaveragefittraindata.prob >= "0.5", 1, 0)
fcreditqualityaveragetrainpred <- factor(creditqualityaveragetrainpred)
levels(fcreditqualityaveragetrainpred) = c('Bad','Good')

#Use "fcreditqualitytrainreal" from logit example as reference (it is the same)

#Creating confusion matrix as a simple overview
ConfMataveragetrain <- confusionMatrix(data=fcreditqualityaveragetrainpred, reference = fcreditqualitytrainreal) #Reference = Real Data

#Display results 
ConfMataveragetrain

#####################################################################################
################ Perform average model.fit on test data #############################
#####################################################################################
creditqualityaveragefittestdata.prob = predict(SavedAveragedModels, creditqualitytest, type="response")

ROUNDcreditqualityaveragefittestdata.prob <- round(creditqualityaveragefittestdata.prob, digits = 3)
creditqualityaveragetestpred <- ifelse(ROUNDcreditqualityaveragefittestdata.prob >= "0.5", 1, 0)
fcreditqualityaveragetestpred <- factor(creditqualityaveragetestpred)
levels(fcreditqualityaveragetestpred) = c('Bad','Good')

#Use "fcreditqualitytestreal" from logit example as reference (it is the same)

#Creating confusion matrix
ConfMataveragetest <- confusionMatrix(data=fcreditqualityaveragetestpred, reference = fcreditqualitytestreal)

#Display results 
ConfMataveragetest

#####################################################################################
# Comparison of AVERAGE Confusion Matrix - AVERAGE Insample VS AVERAGE Out-Of-Sample
#####################################################################################
#Plot Confusion Matrix with function "draw_confusion_matrix"_creditquality_test" for CreditQuality, execute all if not already done!
#Exact
draw_confusion_matrix_creditqualityaverage_train(ConfMataveragetrain)
draw_confusion_matrix_creditqualityaverage_test(ConfMataveragetest)
#Plot
par(mfrow=c(1,2)) #2 graphs side by side
fourfoldplot(ConfMataveragetrain$table, main = "Average Traindata")
fourfoldplot(ConfMataveragetest$table, main = "Average Testdata")

#####################################################################################
########################## Compare Coefficient tables ###############################
#####################################################################################
library(jtools)
library(kableExtra)
library(ggstance)
library(broom.mixed)
library(huxtable)
library(texreg)

#direct (not possible in the same table)
screenreg(SavedAveragedModels, custom.model.names = c("Full Averaged Model"))
screenreg(creditqualitylogittraindata.fit, custom.model.names = c("Logit Model"))


#####################################################################################
############ Confusion Matrix - Comparison of Testdata results #####################
#####################################################################################
#Confusion Matrix - Comparison of Traindata results
#Exact
draw_confusion_matrix_creditquality_train(ConfMattrain)
draw_confusion_matrix_creditqualityaverage_train(ConfMataveragetrain)
#Plot
par(mfrow=c(1,2)) #2 graphs side by side
fourfoldplot(ConfMattrain$table, main = "Logit Traindata")
fourfoldplot(ConfMataveragetrain$table, main = "Average Traindata")

#Confusion Matrix - Comparison of Testdata results
#Exact
draw_confusion_matrix_creditquality_test(ConfMattest)
draw_confusion_matrix_creditqualityaverage_test(ConfMataveragetest)
#Plot
par(mfrow=c(1,2)) #2 graphs side by side
fourfoldplot(ConfMattest$table, main = "Logit Testdata")
fourfoldplot(ConfMataveragetest$table, main = "Average Testdata")

#####################################################################################
############ ROC of Performance with Testdata #######################################
#####################################################################################
library(ROCR)

############### Score train data which was predicted with logit.fit ##################
#Take the prediction "banklogitfittestdata.prob"
predlogittrain = prediction(creditqualitylogitfittraindata.prob, creditqualitytrainreal)

#Perform AUC
auclogittrain = performance(predlogittrain,"auc")@y.values[[1]][1]

#plot ROC curve of Logit Testdataprediction
perflogittrain <- performance(predlogittrain,"tpr","fpr")
plot(perflogittrain, colorize=TRUE, cex.main=1,
     main= paste("Logistic Regression Traindata ROC Curve: AUC =", round(auclogittrain,5)))
abline(a=0, b = 1, col='darkorange1')


############### Score train data which was predicted with average.fit ##################
predAtrain = prediction(creditqualityaveragefittraindata.prob, creditqualitytrainreal)

#Perform AUC
aucAtrain = performance(predAtrain,"auc")@y.values[[1]][1]

#plot ROC curve of averaged Traindataprediction
perfAtrain <- performance(predAtrain,"tpr","fpr")
plot(perfAtrain, col="blue",colorize=TRUE , cex.main=1,
     main= paste("Model Averaged Regression Traindata ROC Curve: AUC =", round(aucAtrain,5)))
abline(a=0, b = 1, col='darkorange1')


############### Score test data which was predicted with logit.fit ##################
#Take the prediction "banklogitfittestdata.prob"
predlogittest = prediction(creditqualitylogitfittestdata.prob, creditqualitytestreal)

#Perform AUC
auclogittest = performance(predlogittest,"auc")@y.values[[1]][1]

#plot ROC curve of Logit Testdataprediction
perflogittest <- performance(predlogittest,"tpr","fpr")
plot(perflogittest, colorize=TRUE, cex.main=1,
     main= paste("Logistic Regression Testdata ROC Curve: AUC =", round(auclogittest,5)))
abline(a=0, b = 1, col='darkorange1')


############### Score test data which was predicted with average.fit ##################
predAtest = prediction(creditqualityaveragefittestdata.prob, creditqualitytestreal)

#Perform AUC
aucAtest = performance(predAtest,"auc")@y.values[[1]][1]

#plot ROC curve of averaged Testdataprediction
perfAtest <- performance(predAtest,"tpr","fpr")
plot(perfAtest, col="blue",colorize=TRUE , cex.main=1,
     main= paste("Model Averaged Regression Testdata ROC Curve: AUC =", round(aucAtest,5)))
abline(a=0, b = 1, col='darkorange1')

#####################################################################################
############### Distribution of Prediction Differences (Absolute) ###################
#####################################################################################

####Logit CreditQuality Prediction
Logitfit.prob <- data.frame(creditqualitylogitfittestdata.prob)
creditqualityreal <- data.frame(creditqualitytestreal)

LogitProb.diff <- cbind(creditqualityreal,Logitfit.prob)
library(tidyverse)
LogitProb.diff <- LogitProb.diff %>% 
  mutate(Logitdiff = creditqualityreal-Logitfit.prob)
LogitProb.diff <- LogitProb.diff %>% 
  mutate(absDiff = abs(Logitdiff))
#Averaged Deviation of all values
colMeans(LogitProb.diff$absDiff)

# Look at positive values which indicate a classification of 1 (Subscribe)
#Positiv 1 - Subscription
positiv <- subset(LogitProb.diff, creditqualitytestreal!=0)
colMeans(positiv$Logitdiff)

# Look at negative values which indicate a classification of 0 (Not Subscribe)
#Negativ 0 - No Subscription
negativ <- subset(LogitProb.diff, creditqualitytestreal!=1)
colMeans(negativ$Logitdiff)

#check if the results are correct => Yes
colMeans(LogitProb.diff$absDiff)
table(creditqualitytestreal)

#####CreditQuality Average Prediction
Logitfit.probA <- data.frame(creditqualityaveragefittestdata.prob)
creditqualityreal <- data.frame(creditqualitytestreal)

LogitProb.diffA <- cbind(creditqualityreal,Logitfit.probA)
library(tidyverse)
LogitProb.diffA <- LogitProb.diffA %>% 
  mutate(LogitdiffA = creditqualityreal-Logitfit.probA)
LogitProb.diffA <- LogitProb.diffA %>% 
  mutate(absDiffA = abs(LogitdiffA))
#Averaged Deviation of all values
colMeans(LogitProb.diffA$absDiffA)

# Look at positive values which indicate a classification of 1 (Good)
#Positiv 1 - Good
positivA <- subset(LogitProb.diffA, creditqualitytestreal!=0)
colMeans(positivA$LogitdiffA)

# Look at negative values which indicate a classification of 0 (Bad)
#Negativ 0 - Bad
negativA <- subset(LogitProb.diffA, creditqualitytestreal!=1)
colMeans(negativA$LogitdiffA)

#check if the results are correct => Yes
colMeans(LogitProb.diffA$absDiffA)
table(creditqualitytestreal)

### Copy all these values in a table in Master Thesis ###
Deviations <- as.data.frame(cbind(nrow(negativ), colMeans(negativ$Logitdiff), nrow(positiv), colMeans(positiv$Logitdiff), colMeans(LogitProb.diff$absDiff)))
DeviationsA <- as.data.frame(cbind(nrow(negativA), colMeans(negativA$LogitdiffA), nrow(positivA), colMeans(positivA$LogitdiffA), colMeans(LogitProb.diffA$absDiffA)))
DeviationsTable <- rbind(Deviations, DeviationsA)
colnames(DeviationsTable) <- c("N Negatives","Negative", "N Positives", "Positive", "Arithmetic Mean")
rownames(DeviationsTable) <- c("Logit", "Average")
DeviationsTable

############################## Plot the density of the calculated deviations LOGIT ######
DeviationsPlot <- as.data.frame(LogitProb.diff$Logitdiff)
DeviationsPlotA <- as.data.frame(LogitProb.diffA$LogitdiffA)

d <- density(DeviationsPlot$creditqualitytestreal)
dA <- density(DeviationsPlotA$creditqualitytestreal)

par(mfrow=c(1,2)) #2 graphs side by side
h <-hist(DeviationsPlot$creditqualitytestreal, breaks=100, xlim = c(-1,1), ylim = c(0,10), xlab="Deviation",
         main="Histogram Deviations of Predicted Outcome - Logit Test Data")
Meanh <- colMeans(LogitProb.diff$absDiff)
lines(c(Meanh,Meanh), c(0,6), col = "green", lwd = 3)#Mean
lines(d, col="red")

#Legend
legend(-1, 9, 
       c("Logit Deviation Density","Mean"), 
       lty=c(1,1), 
       lwd=c(4,4),
       col=c("red","green"))


############################## Plot the density of the calculated deviations AVERAGE
hA <-hist(DeviationsPlotA$creditqualitytestreal, breaks=100, xlim = c(-1,1), ylim = c(0,10), xlab="Deviation",
          main="Histogram Deviations of Predicted Outcome - Average Test Data")
MeanhA <- colMeans(LogitProb.diffA$absDiffA)
lines(c(MeanhA,MeanhA), c(0,6), col = "green", lwd = 3)#Mean
lines(dA, col="blue")

#Legend
legend(-1, 9, 
       c("Average Deviation Density","Mean"), 
       lty=c(1,1), 
       lwd=c(4,4),
       col=c("blue","green"))


############### Density Comparison in same plot ##############################
par(mfrow=c(1,2)) #2 graphs
plot(d, col="red", type = "l", xlim = c(-1,1), ylim = c(0,1.5), main = "Density Comparison of Deviation - Logit", xlab="Deviation" , ylab="Density")
#Legend
legend(-0.95,1.5, 
       c("Logit Deviation Density"), 
       lty=c(1), 
       lwd=c(4),
       col=c("red"))

plot(dA, col="blue", type = "l", xlim = c(-1,1), ylim = c(0,1.5), main = "Density Comparison of Deviation - Average", xlab="Deviation" , ylab="Density")
#Legend
legend(-0.95,1.5, 
       c("Average Deviation Density"), 
       lty=c(1), 
       lwd=c(4),
       col=c("blue"))

#####################################################################################
######################### Example 5.2: AutoClaim ####################################
#####################################################################################
rm(list = ls())
WorkDir <- ("D:/Dropbox/Bearbeitete Dokumente Uni Basel/22Masterarbeit/Code")
setwd(WorkDir)
options(mc.cores = parallel::detectCores())
#####################################################################################
########################## Load and inspect Dataset #################################
#####################################################################################
library(cplm)
data("AutoClaim", package = "cplm")
#'claim incidence'=> I(CLM_FREQ5 > 0)
str(AutoClaim)
summary(AutoClaim)
table(AutoClaim$CLM_FREQ5==0)

#Create Binary ClaimYes
AutoClaim$ClaimYes <- ifelse(AutoClaim$CLM_FREQ5 >= 1, 1, 0)
AutoClaim <- AutoClaim[,c(30,3,7,8,10,14,15,17,20,21,22,28,25)]#selected (25=maxeduc,26=homevalue,27=savehome,9=bluebook)
AutoClaim <- na.omit(AutoClaim)
table(AutoClaim$ClaimYes)
summary(AutoClaim)

#####################################################################################
############# Loop with different sizes of Traindata and fixed Testdata #############
#####################################################################################
#Duration of both loops on my PC: 60min
################################### LOGIT LOOP ######################################
#Create empty dataframe for loopresults
LoopResults <- data.frame(matrix(0, nrow = 5, ncol = 35))

#Split data into training and test set X Percent
library(lattice)
library(ggplot2)
library(caret)
library(caTools)
for (k in 1:35){ #start of for loop with different k for different sample splits
  set.seed(125)
  sample = sample.split(AutoClaim$ClaimYes, SplitRatio = (.7)) #split into 70% train data and 30% testdata
  AutoClaimtrainfull = subset(AutoClaim, sample == TRUE)
  AutoClaimtest  = subset(AutoClaim, sample == FALSE)
  
  #set.seed(123)
  sample = sample.split(AutoClaimtrainfull$ClaimYes, SplitRatio = ((0.099999999/3.5)*k)) #to get 2% steps
  AutoClaimtrainsubset = subset(AutoClaimtrainfull, sample == TRUE) 
  AutoClaimtestunused  = subset(AutoClaimtrainfull, sample == FALSE)
  
  ##################################################
  #predict function - using the fitted function from training dataset on test dataset
  #logit regression with train data
  AutoClaimlogittraindata.fit = glm(ClaimYes ~ TRAVTIME + CAR_USE + REVOLKED + RETAINED
                                    + MVR_PTS + AGE + INCOME + GENDER
                                    + MARRIED + AREA + MAX_EDUC, control=glm.control(maxit=100), data=AutoClaimtrainsubset, family=binomial)
  summary(AutoClaimlogittraindata.fit)
  
  #Perform logit function from train data on test data
  AutoClaimlogitfittestdata.prob = predict(AutoClaimlogittraindata.fit, AutoClaimtest, type="response")
  
  #Round and prepare predicted results
  ROUNDAutoClaimlogitfittestdata.prob <- round(AutoClaimlogitfittestdata.prob, digits = 3)
  AutoClaimtestpred <- ifelse(ROUNDAutoClaimlogitfittestdata.prob >= "0.5", 1, 0)
  fAutoClaimtestpred <- factor(AutoClaimtestpred)
  levels(fAutoClaimtestpred) = c('NoClaim','Claim')
  
  #Prepare real results
  AutoClaimtestreal <- AutoClaimtest[,1]
  AutoClaimtestreal <- ifelse(AutoClaimtestreal >= "0.5", 1, 0)
  fAutoClaimtestreal <- factor(AutoClaimtestreal)
  levels(fAutoClaimtestreal) = c('NoClaim','Claim')
  
  
  #Creating confusion matrix
  ConfMat <- confusionMatrix(data=fAutoClaimtestpred, reference = fAutoClaimtestreal)
  
  #Display results 
  ConfMat
  
  #convert matrix into vector
  ConfMatResultsvector <- as.vector(t(ConfMat$table))
  ConfMatResults <- data.frame(ConfMatResultsvector)
  ConfMatResults #Just do the collected vectors in a data frame?
  ConfMatResults <- rbind(ConfMatResults, k*2)#2% steps
  
  LoopResults[1:5,k] <- data.frame(ConfMatResults) #Do not change, even if you split into more than K=10 datasets!
  #End of for-loop
}

#EXTRACT AND PLOT
ConfMatResultsComplete <- t(LoopResults)


colnames(ConfMatResultsComplete) <- c("A_true_positive", "B_false_positive", "C_false_negative", "D_true_negative", "Size_Training_data_percent")
ConfMatResultsComplete

ConfMatResultsComplete <- data.frame(ConfMatResultsComplete)


#USE ACCURACY, SENSITIVITY AND SPECIFITY INSTEAD
library(tidyverse)
ConfMatResultsComplete <- ConfMatResultsComplete %>% 
  mutate(sample_size = A_true_positive + B_false_positive + C_false_negative + D_true_negative)

ConfMatResultsComplete <- ConfMatResultsComplete %>% 
  mutate(Accuracy = (A_true_positive + D_true_negative)/sample_size)

ConfMatResultsComplete <- ConfMatResultsComplete %>% 
  mutate(sensitivity_recall_TPR = (A_true_positive/(A_true_positive + C_false_negative)))

ConfMatResultsComplete <- ConfMatResultsComplete %>% 
  mutate(specificity_selectivity_TNR = (D_true_negative/(D_true_negative + B_false_positive)))

ConfMatResultsComplete <- ConfMatResultsComplete %>% 
  mutate(FPR = (1-specificity_selectivity_TNR))

################################### AVERAGE LOOP ######################################

#Create empty dataframe for Averageloopresults
LoopaverageResults <- data.frame(matrix(0, nrow = 5, ncol = 35))
library(lattice)
library(ggplot2)
library(caret)
library(caTools)
library(tictoc)
for (k in 1:35){ #start of for loop with different k for different sample splits
  tic()
  set.seed(125)
  sample = sample.split(AutoClaim$ClaimYes, SplitRatio = (.7)) #split into 30% train data and 70% testdata
  AutoClaimaveragetrainfull = subset(AutoClaim, sample == TRUE)
  AutoClaimaveragetest  = subset(AutoClaim, sample == FALSE)
  
  sample = sample.split(AutoClaimaveragetrainfull$ClaimYes, SplitRatio = ((0.099999999/3.5)*k))
  AutoClaimaveragetrainsubset = subset(AutoClaimaveragetrainfull, sample == TRUE)
  AutoClaimaveragetestunused  = subset(AutoClaimaveragetrainfull, sample == FALSE)
  ##################################################
  #predict function - using the fitted function from training dataset on test dataset
  library(MuMIn)
  fm1 <- glm(ClaimYes ~ TRAVTIME + CAR_USE + REVOLKED + RETAINED
             + MVR_PTS + AGE + INCOME + GENDER
             + MARRIED + AREA + MAX_EDUC, data=AutoClaimaveragetrainsubset, family=binomial, control=glm.control(maxit=20), na.action = na.fail)
  #Normal fitting, but cant use it for predict function in this form
  #First step for using predict function
  ####tic()
  ms11 <- lapply(dredge(fm1, evaluate = FALSE), eval)
  ###toc()
  
  ##tic()
  SavedAveragedModels <- model.avg(ms11, subset = delta < 100)
  #toc()
  SavedAveragedModels
  
  #Perform on test data
  AutoClaimaveragelogitfittestdata.prob = predict(SavedAveragedModels, AutoClaimaveragetest, type="response")
  
  #Round and prepare predicted results
  ROUNDAutoClaimaveragelogitfittestdata.prob <- round(AutoClaimaveragelogitfittestdata.prob, digits = 3)
  AutoClaimaveragetestpred <- ifelse(ROUNDAutoClaimaveragelogitfittestdata.prob >= "0.5", 1, 0)
  fAutoClaimaveragetestpred <- factor(AutoClaimaveragetestpred)
  levels(fAutoClaimaveragetestpred) = c('NoClaim','Claim')
  
  #Prepare real results
  AutoClaimaveragetestreal <- AutoClaimaveragetest[,1]
  AutoClaimaveragetestreal <- ifelse(AutoClaimaveragetestreal >= "0.5", 1, 0)
  fAutoClaimaveragetestreal <- factor(AutoClaimaveragetestreal)
  levels(fAutoClaimaveragetestreal) = c('NoClaim','Claim')
  
  #Creating confusion matrix
  ConfMataverage <- confusionMatrix(data=fAutoClaimaveragetestpred, reference = fAutoClaimaveragetestreal)
  
  #Display results 
  ConfMataverage
  
  #convert matrix into vector
  ConfMataverageResultsvector <- as.vector(t(ConfMataverage$table))
  ConfMataverageResults <- data.frame(ConfMataverageResultsvector)
  ConfMataverageResults #Just do the collected vectors in a data frame
  ConfMataverageResults <- rbind(ConfMataverageResults, k*2) #10 for 10%, 5 for 5%
  ConfMataverageResults
  
  LoopaverageResults[1:5,k] <- data.frame(ConfMataverageResults) #No need for a empty dataframe because we can change it from a matrix
  #End of for-loop
  toc()
}

#EXTRACT AND PLOT
#ConfMatResults <- data.frame(ConfMatResults)
ConfMataverageResultsComplete <- t(LoopaverageResults)

colnames(ConfMataverageResultsComplete) <- c("A_true_positive", "B_false_positive", "C_false_negative", "D_true_negative", "Size_Training_data_percent")
ConfMataverageResultsComplete

ConfMataverageResultsComplete <- data.frame(ConfMataverageResultsComplete)

#USE ACCURACY, SENSITIVITY AND SPECIFITY INSTEAD
library(tidyverse)
ConfMataverageResultsComplete <- ConfMataverageResultsComplete %>% 
  mutate(sample_size = A_true_positive + B_false_positive + C_false_negative + D_true_negative)

ConfMataverageResultsComplete <- ConfMataverageResultsComplete %>% 
  mutate(Accuracy = (A_true_positive + D_true_negative)/sample_size)

ConfMataverageResultsComplete <- ConfMataverageResultsComplete %>% 
  mutate(sensitivity_recall_TPR = (A_true_positive/(A_true_positive + C_false_negative)))

ConfMataverageResultsComplete <- ConfMataverageResultsComplete %>% 
  mutate(specificity_selectivity_TNR = (D_true_negative/(D_true_negative + B_false_positive)))

#####################################################################################
####################### Comparison of Learning Curve ################################
#####################################################################################

####################### Logit Relative Plot #########################################
library(ggplot2)
library(graphics)
#Plot Accuracy
par(mfrow=c(2,2)) #1 graph on top and the other below
plot(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$Accuracy, main="AutoClaim Logit Accuracy",
     xlab="Size of training sample in %", ylab="Accuracy",type="o", col="red", pch="o", lty=1, ylim=c(0.56,0.9))

#Plot Sensitivity_Recall_TPR
points(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$sensitivity_recall_TPR, col="blue", pch="+")
lines(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$sensitivity_recall_TPR, col="blue",lty=2)

#Plot Specificity_TNR
points(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$specificity_selectivity_TNR, col="dark blue",pch="*")
lines(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$specificity_selectivity_TNR, col="dark blue", lty=3)

#legend
legend(7,0.74,legend=c("Accuracy","TPR_Sensitivity_Recall","TNR_Specificity"), col=c("red","blue","dark blue"),pch=c("o","+","*"),lty=c(1,2,3), ncol=1)

##################### Average Relative Plot #################################
#Plot Accuracy
plot(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$Accuracy, main="AutoClaim Average Logit Accuracy",
     xlab="Size of training sample in %", ylab="Accuracy", type="o", col="red", pch="o", lty=1, ylim=c(0.56,0.9) )

#Plot Sensitivity_Recall_TPR
points(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$sensitivity_recall_TPR, col="blue", pch="+")
lines(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$sensitivity_recall_TPR, col="blue",lty=2)

#Plot Specificity_TNR
points(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$specificity_selectivity_TNR, col="dark blue",pch="*")
lines(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$specificity_selectivity_TNR, col="dark blue", lty=3)

#legend
legend(7,0.74,legend=c("Accuracy","TPR_Sensitivity_Recall","TNR_Specificity"), col=c("red","blue","dark blue"),pch=c("o","+","*"),lty=c(1,2,3), ncol=1)

########## Logit Plot all absolute values of confusion matrix###############
#Plot A_true_positive
plot(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$A_true_positive, main="AutoClaim Logit Absolute",
     xlab="Size of training sample in %", ylab="Number of absolute Values",type="o", col="red", pch="o", lty=1, ylim=c(200, 1600))

#Plot C_false_negative
points(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$C_false_negative, col="blue", pch="*")
lines(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$C_false_negative, col="blue",lty=2)

#Plot D_true_negative
points(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$D_true_negative, col="dark blue",pch="o")
lines(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$D_true_negative, col="dark blue", lty=3)

#Plot B_false_positive
points(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$B_false_positive, col="dark green",pch="+")
lines(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$B_false_positive, col="dark green", lty=3)

#legend
legend(7,1200,legend=c("A_true_positive", "D_true_negative", "C_false_negative", "B_false_positive"), col=c("red","dark blue","blue","dark green"),pch=c("o","o","*","+"),lty=c(1,2,3), ncol=1)

######### Average Plot all absolute values of confusion matrix ##################
#Plot A_true_positive
plot(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$A_true_positive, main="AutoClaim Average Logit Absolute",
     xlab="Size of training sample in %", ylab="Number of absolute Values",type="o", col="red", pch="o", lty=1, ylim=c(200, 1600))

#Plot C_false_negative
points(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$C_false_negative, col="blue", pch="*")
lines(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$C_false_negative, col="blue",lty=2)

#Plot D_true_negative
points(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$D_true_negative, col="dark blue",pch="o")
lines(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$D_true_negative, col="dark blue", lty=3)

#Plot B_false_positive
points(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$B_false_positive, col="dark green",pch="+")
lines(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$B_false_positive, col="dark green", lty=3)

#legend
legend(7,1200,legend=c("A_true_positive", "D_true_negative", "C_false_negative", "B_false_positive"), col=c("red","dark blue","blue","dark green"),pch=c("o","o","*","+"),lty=c(1,2,3), ncol=1)

#####################################################################################
### Split dataset - 30% train data (Insample) und 70% test data (Out-of-sample) #####
#####################################################################################
library(caTools)
set.seed(125)
sample = sample.split(AutoClaim$ClaimYes, SplitRatio = (.7))
AutoClaimtrainfull = subset(AutoClaim, sample == TRUE)
AutoClaimtest  = subset(AutoClaim, sample == FALSE)

#####################################################################################
############################### Create Logit model ##################################
#####################################################################################
#logit regression with train data
AutoClaimlogittraindata.fit = glm(ClaimYes ~ TRAVTIME + CAR_USE + RETAINED + REVOLKED
                                  + MVR_PTS + AGE + INCOME + GENDER
                                  + MARRIED + AREA + MAX_EDUC, control=glm.control(maxit=100), data=AutoClaimtrainfull, family=binomial)
#Output and Coefficients
summary(AutoClaimlogittraindata.fit)

library(jtools)
library(kableExtra)
library(ggstance)
library(broom.mixed)
library(huxtable)
library(texreg)
#Nice summary
summ(AutoClaimlogittraindata.fit) #clear overview

#Additional ways to show coefficients
#export_summs(AutoClaimlogittraindata.fit, scale = TRUE) #for combining models in 1 table
plot_summs(AutoClaimlogittraindata.fit, scale = TRUE) #coefficient comparison (just best AIC and full model, average model not compatible!)

#Additional ways to show proper output
#screenreg(AutoClaimlogittraindata.fit) #used for comparison later
#huxtablereg(AutoClaimlogittraindata.fit)

#####################################################################################
######################### Perform logit.fit on train data ###########################
#####################################################################################
AutoClaimlogitfittraindata.prob = predict(AutoClaimlogittraindata.fit, AutoClaimtrainfull, type="response")

#Round and shape the predicted probabilities
ROUNDAutoClaimlogitfittraindata.prob <- round(AutoClaimlogitfittraindata.prob, digits = 3)
AutoClaimtrainpred <- ifelse(ROUNDAutoClaimlogitfittraindata.prob >= "0.5", 1, 0)
fAutoClaimtrainpred <- factor(AutoClaimtrainpred)
levels(fAutoClaimtrainpred) = c('NoClaim','Claim')

#Prepare the real outcomes
AutoClaimtrainreal <- AutoClaimtrainfull[,1]
fAutoClaimtrainreal <- factor(AutoClaimtrainreal)
levels(fAutoClaimtrainreal) = c('NoClaim','Claim')

#Creating confusion matrix - how performs the model on the train data - Insample
library(lattice)
library(ggplot2)
library(caret)
ConfMattrain <- confusionMatrix(data=fAutoClaimtrainpred, reference = fAutoClaimtrainreal)#Reference = Real data

#Display results in simple form
ConfMattrain

#####################################################################################
####################### Perform logit model on test data ############################
#####################################################################################
#Perform logit.fit on train data on test data
AutoClaimlogitfittestdata.prob = predict(AutoClaimlogittraindata.fit, AutoClaimtest, type="response")

#Round and prepare predicted probabilities
ROUNDAutoClaimlogitfittestdata.prob <- round(AutoClaimlogitfittestdata.prob, digits = 3)
AutoClaimtestpred <- ifelse(ROUNDAutoClaimlogitfittestdata.prob >= "0.5", 1, 0)
fAutoClaimtestpred <- factor(AutoClaimtestpred)
levels(fAutoClaimtestpred) = c('NoClaim','Claim')

AutoClaimtestreal <- AutoClaimtest[,1]
fAutoClaimtestreal <- factor(AutoClaimtestreal)
levels(fAutoClaimtestreal) = c('NoClaim','Claim')

#Creating confusion matrix - how performs the model on the test data - Out-of-sample
ConfMattest <- confusionMatrix(data=fAutoClaimtestpred, reference = fAutoClaimtestreal)

#Display results 
ConfMattest

#####################################################################################
#Compare Confusion Matrix of Traindata and Testdata - Logit Insample VS Out-Of-Sample
#####################################################################################
#Plot Confusion Matrix with function "draw_confusion_matrix" for Autoclaim - execute all if not already done!
#Exact
draw_confusion_matrix_AutoClaim_train(ConfMattrain)
draw_confusion_matrix_AutoClaim_test(ConfMattest)
#Plot
par(mfrow=c(1,2)) #2 graphs side by side
fourfoldplot(ConfMattrain$table, main = "Logit Traindata")
fourfoldplot(ConfMattest$table, main = "Logit Testdata")


#####################################################################################
######################### Perform Average Method ####################################
#####################################################################################
library(MuMIn)

AutoClaim1 <- glm(ClaimYes ~ TRAVTIME + CAR_USE + RETAINED + REVOLKED
                  + MVR_PTS + AGE + INCOME + GENDER
                  + MARRIED + AREA + MAX_EDUC, data=AutoClaimtrainfull, family=binomial, control=glm.control(maxit=100), na.action = na.fail)

#####################################################################################
################## MODEL INSPECTION WITH DIFFERENT COMMANDS #########################
#####################################################################################
#Normal fitting, but cant use it for predict function in this form
#model selection: you can combine models manually(model.sel) or automatically over "dredge"
(msAutoClaim1 <- dredge(AutoClaim1, rank = "AICc", extra = c("R^2","adjR^2", F = function(x) #Create Model Selection table
  summary(x)$fstatistik[[1]])))
msAutoClaim1 #Show model selection table
msAutoClaim1[1:4] #show model selection table but new weights!

#Plot a graph of all models and their weights
par(mfrow=c(1,1)) #Plot just 1 graph
par(oma = c(0,1,2,2)) #increase distance to window borders to prevent cutted plot
plot(msAutoClaim1) #Make plot window much bigger to plot without cutting

#After creating a model selection table, you can use "model.avg" to choose a subset of models
msAutoClaim11 <- model.avg(msAutoClaim1, subset = delta < 3) #Averaging the models with subset = delta < 1
msAutoClaim11

#Show coefficients of full model with CI
msAutoClaimfull <- cbind(summary(msAutoClaim11)$coefmat.full, confint(msAutoClaim11, full = TRUE))
round(msAutoClaimfull, digits = 8)

#Show coefficients of full model with CI
msAutoClaimsubset <- cbind(summary(msAutoClaim11)$coefmat.subset, confint(msAutoClaim11, full = FALSE))
round(msAutoClaimsubset, digits = 8)

#Show the best model with best AICc
get.models(msAutoClaim1, 1)#Object after dredge

#Subset best 4 models
msAutoClaim1Subset <- msAutoClaim1[1:5, ]
screenreg(msAutoClaim1Subset, custom.model.names = c("Best Model", "2. Best Model", "3. Best Model", "4. Best Model", "5. Best Model"))#show best 5 models

#####################################################################################
##################### Perform averaged model for Prediction #########################
#####################################################################################
#First step for using predict function:
library(tictoc)
tic()
#Create List of evaluated models ordered by AICc
AutoClaim11 <- lapply(dredge(AutoClaim1, rank = "AICc", evaluate = FALSE), eval)
toc()

tic()
#Average the models with a choosen set of models
SavedAveragedModels <- model.avg(AutoClaim11, subset = delta < 100)
toc()
SavedAveragedModels

#####################################################################################
#################### Perform average model.fit on train data ########################
#####################################################################################
AutoClaimaveragefittraindata.prob = predict(SavedAveragedModels, AutoClaimtrainfull, type="response")

ROUNDAutoClaimaveragefittraindata.prob <- round(AutoClaimaveragefittraindata.prob, digits = 3)
AutoClaimaveragetrainpred <- ifelse(ROUNDAutoClaimaveragefittraindata.prob >= "0.5", 1, 0)
fAutoClaimaveragetrainpred <- factor(AutoClaimaveragetrainpred)
levels(fAutoClaimaveragetrainpred) = c('NoClaim','Claim')

#Use "fAutoClaimtrainreal" from logit example as reference (it is the same)

#Creating confusion matrix as a simple overview
ConfMataveragetrain <- confusionMatrix(data=fAutoClaimaveragetrainpred, reference = fAutoClaimtrainreal) #Reference = Real Data

#Display results 
ConfMataveragetrain

#####################################################################################
################ Perform average model.fit on test data #############################
#####################################################################################
AutoClaimaveragefittestdata.prob = predict(SavedAveragedModels, AutoClaimtest, type="response")

ROUNDAutoClaimaveragefittestdata.prob <- round(AutoClaimaveragefittestdata.prob, digits = 3)
AutoClaimaveragetestpred <- ifelse(ROUNDAutoClaimaveragefittestdata.prob >= "0.5", 1, 0)
fAutoClaimaveragetestpred <- factor(AutoClaimaveragetestpred)
levels(fAutoClaimaveragetestpred) = c('NoClaim','Claim')

#Use "fAutoClaimtestreal" from logit example as reference (it is the same)

#Creating confusion matrix
ConfMataveragetest <- confusionMatrix(data=fAutoClaimaveragetestpred, reference = fAutoClaimtestreal)

#Display results 
ConfMataveragetest

#####################################################################################
# Comparison of AVERAGE Confusion Matrix - AVERAGE Insample VS AVERAGE Out-Of-Sample
#####################################################################################

#Exact
draw_confusion_matrix_AutoClaimaverage_train(ConfMataveragetrain)
draw_confusion_matrix_AutoClaimaverage_test(ConfMataveragetest)
#Plot
par(mfrow=c(1,2)) #2 graphs side by side
fourfoldplot(ConfMataveragetrain$table, main = "Average Traindata")
fourfoldplot(ConfMataveragetest$table, main = "Average Testdata")

#####################################################################################
########################## Compare Coefficient tables ###############################
#####################################################################################
library(jtools)
library(kableExtra)
library(ggstance)
library(broom.mixed)
library(huxtable)
library(texreg)

#direct (not possible in the same table)
screenreg(SavedAveragedModels, custom.model.names = c("Full Averaged Model"))
screenreg(AutoClaimlogittraindata.fit, custom.model.names = c("Logit Model"))#not ordered


#####################################################################################
############ Confusion Matrix - Comparison of Testdata results ######################
#####################################################################################
#Confusion Matrix - Comparison of Traindata results
#Exact
draw_confusion_matrix_AutoClaim_train(ConfMattrain)
draw_confusion_matrix_AutoClaimaverage_train(ConfMataveragetrain)
#Plot
par(mfrow=c(1,2)) #2 graphs side by side
fourfoldplot(ConfMattrain$table, main = "Logit Traindata")
fourfoldplot(ConfMataveragetrain$table, main = "Average Traindata")

#Confusion Matrix - Comparison of Testdata results
#Exact
draw_confusion_matrix_AutoClaim_test(ConfMattest)
draw_confusion_matrix_AutoClaimaverage_test(ConfMataveragetest)
#Plot
par(mfrow=c(1,2)) #2 graphs side by side
fourfoldplot(ConfMattest$table, main = "Logit Testdata")
fourfoldplot(ConfMataveragetest$table, main = "Average Testdata")

#####################################################################################
############ ROC of Performance with Testdata #######################################
#####################################################################################
library(ROCR)

############### Score train data which was predicted with logit.fit ##################
#Take the prediction "banklogitfittestdata.prob"
predlogittrain = prediction(AutoClaimlogitfittraindata.prob, AutoClaimtrainreal)

#Perform AUC
auclogittrain = performance(predlogittrain,"auc")@y.values[[1]][1]

#plot ROC curve of Logit Testdataprediction
perflogittrain <- performance(predlogittrain,"tpr","fpr")
plot(perflogittrain, colorize=TRUE, cex.main=1,
     main= paste("Logistic Regression Traindata ROC Curve: AUC =", round(auclogittrain,5)))
abline(a=0, b = 1, col='darkorange1')


############### Score train data which was predicted with average.fit ##################
predAtrain = prediction(AutoClaimaveragefittraindata.prob, AutoClaimtrainreal)

#Perform AUC
aucAtrain = performance(predAtrain,"auc")@y.values[[1]][1]

#plot ROC curve of averaged Traindataprediction
perfAtrain <- performance(predAtrain,"tpr","fpr")
plot(perfAtrain, col="blue",colorize=TRUE , cex.main=1,
     main= paste("Model Averaged Regression Traindata ROC Curve: AUC =", round(aucAtrain,5)))
abline(a=0, b = 1, col='darkorange1')


############### Score test data which was predicted with logit.fit ##################
#Take the prediction "banklogitfittestdata.prob"
predlogittest = prediction(AutoClaimlogitfittestdata.prob, AutoClaimtestreal)

#Perform AUC
auclogittest = performance(predlogittest,"auc")@y.values[[1]][1]

#plot ROC curve of Logit Testdataprediction
perflogittest <- performance(predlogittest,"tpr","fpr")
plot(perflogittest, colorize=TRUE, cex.main=1,
     main= paste("Logistic Regression Testdata ROC Curve: AUC =", round(auclogittest,5)))
abline(a=0, b = 1, col='darkorange1')


############### Score test data which was predicted with average.fit ##################
predAtest = prediction(AutoClaimaveragefittestdata.prob, AutoClaimtestreal)

#Perform AUC
aucAtest = performance(predAtest,"auc")@y.values[[1]][1]

#plot ROC curve of averaged Testdataprediction
perfAtest <- performance(predAtest,"tpr","fpr")
plot(perfAtest, col="blue",colorize=TRUE , cex.main=1,
     main= paste("Model Averaged Regression Testdata ROC Curve: AUC =", round(aucAtest,5)))
abline(a=0, b = 1, col='darkorange1')

#####################################################################################
############### Distribution of Prediction Differences (Absolute) ###################
#####################################################################################

######Logit AutoClaim Prediction
Logitfit.prob <- data.frame(AutoClaimlogitfittestdata.prob)
AutoClaimreal <- data.frame(AutoClaimtestreal)

LogitProb.diff <- cbind(AutoClaimreal,Logitfit.prob)
library(tidyverse)
LogitProb.diff <- LogitProb.diff %>% 
  mutate(Logitdiff = AutoClaimreal-Logitfit.prob)
LogitProb.diff <- LogitProb.diff %>% 
  mutate(absDiff = abs(Logitdiff))
#Averaged Deviation of all values
colMeans(LogitProb.diff$absDiff)

# Look at positive values which indicate a classification of 1 (Claim)
#Positiv 1 - Claim
positiv <- subset(LogitProb.diff, AutoClaimtestreal!=0)
colMeans(positiv$Logitdiff)

# Look at negative values which indicate a classification of 0 (Not Claim)
#Negativ 0 - No Claim
negativ <- subset(LogitProb.diff, AutoClaimtestreal!=1)
colMeans(negativ$Logitdiff)

#check if the results are correct => Yes
colMeans(LogitProb.diff$absDiff)
table(AutoClaimtestreal)

#######AutoClaim Average Prediction
Logitfit.probA <- data.frame(AutoClaimaveragefittestdata.prob)
AutoClaimreal <- data.frame(AutoClaimtestreal)

LogitProb.diffA <- cbind(AutoClaimreal,Logitfit.probA)
library(tidyverse)
LogitProb.diffA <- LogitProb.diffA %>% 
  mutate(LogitdiffA = AutoClaimreal-Logitfit.probA)
LogitProb.diffA <- LogitProb.diffA %>% 
  mutate(absDiffA = abs(LogitdiffA))
#Averaged Deviation of all values
colMeans(LogitProb.diffA$absDiffA)

# Look at positive values which indicate a classification of 1 (Claim)
#Positiv 1 - Claim
positivA <- subset(LogitProb.diffA, AutoClaimtestreal!=0)
colMeans(positivA$LogitdiffA)

# Look at negative values which indicate a classification of 0 (Not Claim)
#Negativ 0 - No Claim
negativA <- subset(LogitProb.diffA, AutoClaimtestreal!=1)
colMeans(negativA$LogitdiffA)

#check if the results are correct => Yes
colMeans(LogitProb.diffA$absDiffA)
table(AutoClaimtestreal)

### Copy all these values in a table in Master Thesis ###
Deviations <- as.data.frame(cbind(nrow(negativ), colMeans(negativ$Logitdiff), nrow(positiv), colMeans(positiv$Logitdiff), colMeans(LogitProb.diff$absDiff)))
DeviationsA <- as.data.frame(cbind(nrow(negativA), colMeans(negativA$LogitdiffA), nrow(positivA), colMeans(positivA$LogitdiffA), colMeans(LogitProb.diffA$absDiffA)))
DeviationsTable <- rbind(Deviations, DeviationsA)
colnames(DeviationsTable) <- c("N Negatives","Negative", "N Positives", "Positive", "Arithmetic Mean")
rownames(DeviationsTable) <- c("Logit", "Average")
DeviationsTable

############################## Plot the density of the calculated deviations LOGIT ######
DeviationsPlot <- as.data.frame(LogitProb.diff$Logitdiff)
DeviationsPlotA <- as.data.frame(LogitProb.diffA$LogitdiffA)

d <- density(DeviationsPlot$AutoClaimtestreal)
dA <- density(DeviationsPlotA$AutoClaimtestreal)

par(mfrow=c(1,2)) #2 graphs side by side
h <-hist(DeviationsPlot$AutoClaimtestreal, breaks=1000, xlim = c(-1,1), ylim = c(0,28), xlab="Deviation",
         main="Histogram Deviations - Logit Test Data")
Meanh <- colMeans(LogitProb.diff$absDiff)
lines(c(Meanh,Meanh), c(0,12), col = "green", lwd = 3)#Mean
lines(d, col="red")

#Legend
legend(-0.03, 25, 
       c("Logit Deviation Density","Mean"), 
       lty=c(1,1), 
       lwd=c(4,4),
       col=c("red","green"))


############################## Plot the density of the calculated deviations AVERAGE
hA <-hist(DeviationsPlotA$AutoClaimtestreal, breaks=1000, xlim = c(-1,1), ylim = c(0,28), xlab="Deviation",
          main="Histogram Deviations - Average Test Data")
MeanhA <- colMeans(LogitProb.diffA$absDiffA)
lines(c(MeanhA,MeanhA), c(0,12), col = "green", lwd = 3)#Mean
lines(dA, col="blue")

#Legend
legend(-0.04, 25, 
       c("Average Deviation Density","Mean"), 
       lty=c(1,1), 
       lwd=c(4,4),
       col=c("blue","green"))


############### Density Comparison in same plot ##############################
par(mfrow=c(1,2)) #2 graphs
plot(d, col="red", type = "l", xlim = c(-1,1), ylim = c(0,1.4), main = "Density Comparison of Deviation - Logit", xlab="Deviation" , ylab="Density")
#Legend
legend(-0.05,1.2, 
       c("Logit Deviation Density"), 
       lty=c(1), 
       lwd=c(4),
       col=c("red"))

plot(dA, col="blue", type = "l", xlim = c(-1,1), ylim = c(0,1.4), main = "Density Comparison of Deviation - Average", xlab="Deviation" , ylab="Density")
#Legend
legend(-0.05,1.2, 
       c("Average Deviation Density"), 
       lty=c(1), 
       lwd=c(4),
       col=c("blue"))

#############################################################################
##################### Example 5.3: Wine Data ################################
#############################################################################
rm(list = ls())
WorkDir <- ("D:/Dropbox/Bearbeitete Dokumente Uni Basel/22Masterarbeit/Code")
setwd(WorkDir)
options(mc.cores = parallel::detectCores())
# White wine
white <- read.csv2("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", dec = ".", header = TRUE) 
library(tidyverse)
white$quality <- ifelse(white$quality >= "7", 1, 0)
wine <- na.omit(white)
rm(list = c("white"))

# Overview: 
summary(wine)
table(wine$quality) #0 = badquality, 1 = goodquality
#####################################################################################
############# Loop with different sizes of Traindata and fixed Testdata #############
#####################################################################################
#Duration of both loops: 50min
################################### LOGIT LOOP ######################################
#Create empty dataframe for loopresults
LoopResults <- data.frame(matrix(0, nrow = 5, ncol = 90))

#Split data into training 90% and test set 10 Percent
library(lattice)
library(ggplot2)
library(caret)
library(caTools)
for (k in 1:90){ #start of for loop with different k for different sample splits
  set.seed(337)
  sample = sample.split(wine$quality, SplitRatio = (.9)) #split into 90% train data and 10% testdata
  winetrainfull = subset(wine, sample == TRUE)
  winetest  = subset(wine, sample == FALSE)
  
  sample = sample.split(winetrainfull$quality, SplitRatio = ((0.099999999/9)*k)) # split traindata again to get 1% steps
  winetrainsubset = subset(winetrainfull, sample == TRUE) #This leads to traindatasets of 1%, 2%, 3%, 4%.....90%
  winetestunused  = subset(winetrainfull, sample == FALSE)
  
  ##################################################
  #predict function - using the fitted function from training dataset on test dataset
  #logit regression with train data
  winelogittraindata.fit = glm(quality ~ fixed.acidity + volatile.acidity + 
                                 citric.acid + residual.sugar + chlorides + density +
                                 free.sulfur.dioxide + total.sulfur.dioxide + pH + sulphates + alcohol, control=glm.control(maxit=100), data=winetrainsubset, family=binomial)
  summary(winelogittraindata.fit)
  
  #Perform logit function from train data on test data
  winelogitfittestdata.prob = predict(winelogittraindata.fit, winetest, type="response")
  
  #Round and prepare predicted results
  ROUNDwinelogitfittestdata.prob <- round(winelogitfittestdata.prob, digits = 3)
  winetestpred <- ifelse(ROUNDwinelogitfittestdata.prob >= "0.5", 1, 0)
  fwinetestpred <- factor(winetestpred)
  levels(fwinetestpred) = c('Bad Quality','Good Quality')
  
  #Prepare real results
  winetestreal <- winetest[,12]
  winetestreal <- ifelse(winetestreal >= "0.5", 1, 0)
  fwinetestreal <- factor(winetestreal)
  levels(fwinetestreal) = c('Bad Quality','Good Quality')
  
  
  #Creating confusion matrix
  ConfMat <- confusionMatrix(data=fwinetestpred, reference = fwinetestreal)
  
  #Display results 
  ConfMat
  
  #convert matrix into vector
  ConfMatResultsvector <- as.vector(t(ConfMat$table))
  ConfMatResults <- data.frame(ConfMatResultsvector)
  ConfMatResults #Just do the collected vectors in a data frame?
  ConfMatResults <- rbind(ConfMatResults, k*1)#1% steps
  
  LoopResults[1:5,k] <- data.frame(ConfMatResults) #Do not change, even if you split into more than K=10 datasets!
  #End of for-loop
}

#EXTRACT AND PLOT
ConfMatResultsComplete <- t(LoopResults)


colnames(ConfMatResultsComplete) <- c("A_true_positive", "B_false_positive", "C_false_negative", "D_true_negative", "Size_Training_data_percent")
ConfMatResultsComplete

ConfMatResultsComplete <- data.frame(ConfMatResultsComplete)


#USE ACCURACY, SENSITIVITY AND SPECIFITY INSTEAD
library(tidyverse)
ConfMatResultsComplete <- ConfMatResultsComplete %>% 
  mutate(sample_size = A_true_positive + B_false_positive + C_false_negative + D_true_negative)

ConfMatResultsComplete <- ConfMatResultsComplete %>% 
  mutate(Accuracy = (A_true_positive + D_true_negative)/sample_size)

ConfMatResultsComplete <- ConfMatResultsComplete %>% 
  mutate(sensitivity_recall_TPR = (A_true_positive/(A_true_positive + C_false_negative)))

ConfMatResultsComplete <- ConfMatResultsComplete %>% 
  mutate(specificity_selectivity_TNR = (D_true_negative/(D_true_negative + B_false_positive)))

ConfMatResultsComplete <- ConfMatResultsComplete %>% 
  mutate(FPR = (1-specificity_selectivity_TNR))

################################### AVERAGE LOOP ######################################

#Create empty dataframe for Averageloopresults
LoopaverageResults <- data.frame(matrix(0, nrow = 5, ncol = 90))
library(lattice)
library(ggplot2)
library(caret)
library(caTools)
library(tictoc)
for (k in 1:90){ #start of for loop with different k for different sample splits
  tic()
  set.seed(337)
  sample = sample.split(wine$quality, SplitRatio = (.9)) #split into 90% train data and 10% testdata
  wineaveragetrainfull = subset(wine, sample == TRUE)
  wineaveragetest  = subset(wine, sample == FALSE)
  
  sample = sample.split(wineaveragetrainfull$quality, SplitRatio = ((0.099999999/9)*k)) # split traindata again
  wineaveragetrainsubset = subset(wineaveragetrainfull, sample == TRUE) #This leads to traindatasets of 1%, 2%, 3%, 4%...90%
  wineaveragetestunused  = subset(wineaveragetrainfull, sample == FALSE)
  ##################################################
  #predict function - using the fitted function from training dataset on test dataset
  library(MuMIn)
  fm1 <- glm(quality ~ fixed.acidity + volatile.acidity + 
               citric.acid + residual.sugar + chlorides + 
               free.sulfur.dioxide + total.sulfur.dioxide + density
             + pH + sulphates + alcohol, data=wineaveragetrainsubset, family=binomial, control=glm.control(maxit=100), na.action = na.fail)
  #First step for using predict function
  ####tic()
  ms11 <- lapply(dredge(fm1, evaluate = FALSE), eval)
  ###toc()
  
  ##tic()
  SavedAveragedModels <- model.avg(ms11, subset = delta < 100)
  #toc()
  SavedAveragedModels
  
  #Perform on test data
  wineaveragelogitfittestdata.prob = predict(SavedAveragedModels, wineaveragetest, type="response")
  
  #Round and prepare predicted results
  ROUNDwineaveragelogitfittestdata.prob <- round(wineaveragelogitfittestdata.prob, digits = 3)
  wineaveragetestpred <- ifelse(ROUNDwineaveragelogitfittestdata.prob >= "0.5", 1, 0)
  fwineaveragetestpred <- factor(wineaveragetestpred)
  levels(fwineaveragetestpred) = c('Bad Quality','Good Quality')
  
  #Prepare real results
  wineaveragetestreal <- wineaveragetest[,12]
  wineaveragetestreal <- ifelse(wineaveragetestreal >= "0.5", 1, 0)
  fwineaveragetestreal <- factor(wineaveragetestreal)
  levels(fwineaveragetestreal) = c('Bad Quality','Good Quality')
  
  #Creating confusion matrix
  ConfMataverage <- confusionMatrix(data=fwineaveragetestpred, reference = fwineaveragetestreal)
  
  #Display results 
  ConfMataverage
  
  #convert matrix into vector
  ConfMataverageResultsvector <- as.vector(t(ConfMataverage$table))
  ConfMataverageResults <- data.frame(ConfMataverageResultsvector)
  ConfMataverageResults #Just do the collected vectors in a data frame
  ConfMataverageResults <- rbind(ConfMataverageResults, k*1) #10 for 10%, 5 for 5%
  ConfMataverageResults
  
  LoopaverageResults[1:5,k] <- data.frame(ConfMataverageResults) #No need for a empty dataframe because we can change it from a matrix
  #End of for-loop
  toc()
}

#EXTRACT AND PLOT
#ConfMatResults <- data.frame(ConfMatResults)
ConfMataverageResultsComplete <- t(LoopaverageResults)

colnames(ConfMataverageResultsComplete) <- c("A_true_positive", "B_false_positive", "C_false_negative", "D_true_negative", "Size_Training_data_percent")
ConfMataverageResultsComplete

ConfMataverageResultsComplete <- data.frame(ConfMataverageResultsComplete)

#USE ACCURACY, SENSITIVITY AND SPECIFITY INSTEAD
library(tidyverse)
ConfMataverageResultsComplete <- ConfMataverageResultsComplete %>% 
  mutate(sample_size = A_true_positive + B_false_positive + C_false_negative + D_true_negative)

ConfMataverageResultsComplete <- ConfMataverageResultsComplete %>% 
  mutate(Accuracy = (A_true_positive + D_true_negative)/sample_size)

ConfMataverageResultsComplete <- ConfMataverageResultsComplete %>% 
  mutate(sensitivity_recall_TPR = (A_true_positive/(A_true_positive + C_false_negative)))

ConfMataverageResultsComplete <- ConfMataverageResultsComplete %>% 
  mutate(specificity_selectivity_TNR = (D_true_negative/(D_true_negative + B_false_positive)))

#####################################################################################
####################### Comparison of Learning Curve ################################
#####################################################################################

####################### Logit Relative Plot #########################################
library(ggplot2)
library(graphics)
#Plot Accuracy
par(mfrow=c(2,2)) #1 graph on top and the other below
plot(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$Accuracy, main="WineQuality Logit Accuracy",
     xlab="Size of training sample in %", ylab="Accuracy",type="o", col="red", pch="o", lty=1, ylim=c(0.2,1))

#Plot Sensitivity_Recall_TPR
points(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$sensitivity_recall_TPR, col="blue", pch="+")
lines(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$sensitivity_recall_TPR, col="blue",lty=2)

#Plot Specificity_TNR
points(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$specificity_selectivity_TNR, col="dark blue",pch="*")
lines(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$specificity_selectivity_TNR, col="dark blue", lty=3)

#legend
legend(18,0.7,legend=c("Accuracy","TPR_Sensitivity_Recall","TNR_Specificity"), col=c("red","blue","dark blue"),pch=c("o","+","*"),lty=c(1,2,3), ncol=1)

##################### Average Relative Plot #################################
#Plot Accuracy
plot(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$Accuracy, main="WineQuality Average Logit Accuracy",
     xlab="Size of training sample in %", ylab="Accuracy", type="o", col="red", pch="o", lty=1, ylim=c(0.2,1) )

#Plot Sensitivity_Recall_TPR
points(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$sensitivity_recall_TPR, col="blue", pch="+")
lines(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$sensitivity_recall_TPR, col="blue",lty=2)

#Plot Specificity_TNR
points(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$specificity_selectivity_TNR, col="dark blue",pch="*")
lines(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$specificity_selectivity_TNR, col="dark blue", lty=3)

#legend
legend(18,0.7,legend=c("Accuracy","TPR_Sensitivity_Recall","TNR_Specificity"), col=c("red","blue","dark blue"),pch=c("o","+","*"),lty=c(1,2,3), ncol=1)

########## Logit Plot all absolute values of confusion matrix###############
#Plot A_true_positive
plot(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$A_true_positive, main="WineQuality Logit Absolute",
     xlab="Size of training sample in %", ylab="Number of absolute Values",type="o", col="red", pch="o", lty=1, ylim=c(0, 400))

#Plot C_false_negative
points(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$C_false_negative, col="blue", pch="*")
lines(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$C_false_negative, col="blue",lty=2)

#Plot D_true_negative
points(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$D_true_negative, col="dark blue",pch="o")
lines(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$D_true_negative, col="dark blue", lty=3)

#Plot B_false_positive
points(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$B_false_positive, col="dark green",pch="+")
lines(ConfMatResultsComplete$Size_Training_data_percent, ConfMatResultsComplete$B_false_positive, col="dark green", lty=3)

#legend
legend(18,300,legend=c("A_true_positive", "D_true_negative", "C_false_negative", "B_false_positive"), col=c("red","dark blue","blue","dark green"),pch=c("o","o","*","+"),lty=c(1,2,3), ncol=1)

######### Average Plot all absolute values of confusion matrix ##################
#Plot A_true_positive
plot(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$A_true_positive, main="WineQuality Average Logit Absolute",
     xlab="Size of training sample in %", ylab="Number of absolute Values",type="o", col="red", pch="o", lty=1, ylim=c(0, 400))

#Plot C_false_negative
points(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$C_false_negative, col="blue", pch="*")
lines(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$C_false_negative, col="blue",lty=2)

#Plot D_true_negative
points(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$D_true_negative, col="dark blue",pch="o")
lines(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$D_true_negative, col="dark blue", lty=3)

#Plot B_false_positive
points(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$B_false_positive, col="dark green",pch="+")
lines(ConfMataverageResultsComplete$Size_Training_data_percent, ConfMataverageResultsComplete$B_false_positive, col="dark green", lty=3)

#legend
legend(18,300,legend=c("A_true_positive", "D_true_negative", "C_false_negative", "B_false_positive"), col=c("red","dark blue","blue","dark green"),pch=c("o","o","*","+"),lty=c(1,2,3), ncol=1)

##################################################
#Datensatz teilen - 90% train data (Insample) und 10% test data (Out-of-sample)
##################################################
library(caTools)
set.seed(337)
sample = sample.split(wine$quality, SplitRatio = (.9))
winetrainfull = subset(wine, sample == TRUE)
winetest  = subset(wine, sample == FALSE)

#####################################################################################
############################### Create Logit model ##################################
#####################################################################################
#logit regression with train data
winelogittraindata.fit = glm(quality ~ fixed.acidity + volatile.acidity + 
                                    citric.acid + residual.sugar + chlorides + 
                                    free.sulfur.dioxide + total.sulfur.dioxide + 
                                    density + pH + sulphates + alcohol, control=glm.control(maxit=100), data=winetrainfull, family=binomial)
#Output and Coefficients
summary(winelogittraindata.fit)

library(jtools)
library(kableExtra)
library(ggstance)
library(broom.mixed)
library(huxtable)
library(texreg)
#Nice summary
summ(winelogittraindata.fit) #clear overview

#Additional ways to show coefficients
#export_summs(winelogittraindata.fit, scale = TRUE) #for combining models in 1 table

plot_summs(winelogittraindata.fit, scale = TRUE) #coefficient comparison (just best AIC and full model, average model not compatible!)

#Additional ways to show proper output
#screenreg(winelogittraindata.fit) #used for comparison later
#huxtablereg(winelogittraindata.fit)

#try and error
#install.packages("modelsummary")
#library(modelsummary)
#models <- list()
#models[['Logit']] <- winelogittraindata.fit
#modelsummary(models)

#####################################################################################
######################## Perform logit.fit on train data ############################
#####################################################################################
winelogitfittraindata.prob = predict(winelogittraindata.fit, winetrainfull, type="response")

#Round and shape the predicted probabilities
ROUNDwinelogitfittraindata.prob <- round(winelogitfittraindata.prob, digits = 3)
winetrainpred <- ifelse(ROUNDwinelogitfittraindata.prob >= "0.5", 1, 0)
fwinetrainpred <- factor(winetrainpred)
levels(fwinetrainpred) = c('Bad Quality','Good Quality')

#Prepare the real outcomes
winetrainreal <- winetrainfull[,12]
fwinetrainreal <- factor(winetrainreal)
levels(fwinetrainreal) = c('Bad Quality','Good Quality')

#Creating confusion matrix - how performs the model on the train data - Insample
library(lattice)
library(ggplot2)
library(caret)
ConfMattrain <- confusionMatrix(data=fwinetrainpred, reference = fwinetrainreal)#Reference = Real data

#Display results in simple form
ConfMattrain

#####################################################################################
####################### Perform logit model on test data ############################
#####################################################################################
#Perform logit.fit on train data on test data
winelogitfittestdata.prob = predict(winelogittraindata.fit, winetest, type="response")

#Round and prepare predicted probabilities
ROUNDwinelogitfittestdata.prob <- round(winelogitfittestdata.prob, digits = 3)
winetestpred <- ifelse(ROUNDwinelogitfittestdata.prob >= "0.5", 1, 0)
fwinetestpred <- factor(winetestpred)
levels(fwinetestpred) = c('Bad Quality','Good Quality')

winetestreal <- winetest[,12]
fwinetestreal <- factor(winetestreal)
levels(fwinetestreal) = c('Bad Quality','Good Quality')

#Creating confusion matrix - how performs the model on the test data - Out-of-sample
ConfMattest <- confusionMatrix(data=fwinetestpred, reference = fwinetestreal)

#Display results 
ConfMattest

#####################################################################################
#Compare Confusion Matrix of Traindata and Testdata - Logit Insample VS Out-Of-Sample
#####################################################################################
#USE FUNCTION "draw_confusion_matrix_all"! Execute all functions for later use!
#Exact
draw_confusion_matrix_wine_train(ConfMattrain)
draw_confusion_matrix_wine_test(ConfMattest)
#Plot
par(mfrow=c(1,2)) #2 graphs side by side
fourfoldplot(ConfMattrain$table, main = "Logit Traindata")
fourfoldplot(ConfMattest$table, main = "Logit Testdata")


#####################################################################################
######################### Perform Average Method ####################################
#####################################################################################
library(MuMIn)
wine1 <- glm(quality ~ fixed.acidity + volatile.acidity + 
               citric.acid + residual.sugar + chlorides + 
               free.sulfur.dioxide + total.sulfur.dioxide + 
               density + pH + sulphates + alcohol, data=winetrainfull, family=binomial, control=glm.control(maxit=100), na.action = na.fail)
#####################################################################################
################## MODEL INSPECTION WITH DIFFERENT COMMANDS #########################
#####################################################################################
#Normal fitting, but cant use it for predict function in this form
#model selection: you can combine models manually(model.sel) or automatically over "dredge"
(mswine1 <- dredge(wine1, rank = "AICc", extra = c("R^2","adjR^2", F = function(x) #Create Model Selection table
  summary(x)$fstatistik[[1]])))
mswine1 #Show model selection table
mswine1[1:4] #show model selection table but new weights!

#Plot a graph of all models and their weights
par(mfrow=c(1,1)) #Plot just 1 graph
par(oma = c(0,1,2,2)) #increase distance to window borders to prevent cutted plot
plot(mswine1) #Make plot window much bigger to plot without cutting

#After creating a model selection table, you can use "model.avg" to choose a subset of models
mswine11 <- model.avg(mswine1, subset = delta < 2) #Averaging the models with subset = delta < 1
mswine11

#Show coefficients of full model with CI
mswinefull <- cbind(summary(mswine11)$coefmat.full, confint(mswine11, full = TRUE))
round(mswinefull, digits = 8)

#Show coefficients of full model with CI
mswinesubset <- cbind(summary(mswine11)$coefmat.subset, confint(mswine11, full = FALSE))
round(mswinesubset, digits = 8)

#Show the best model with best AICc
get.models(mswine1, 1)#Object after dredge

#Subset best 4 models
mswine1Subset <- mswine1[1:4, ]
screenreg(mswine1Subset, custom.model.names = c("Best Model", "2. Best Model", "3. Best Model", "4. Best Model"))#show best 4 models
#####################################################################################
################## Perform averaged logit model on train data to check ##############
#####################################################################################
#First step for using predict function:
library(tictoc)
tic()
#Create List of evaluated models ordered by AIC
wine11 <- lapply(dredge(wine1, rank = "AICc", evaluate = FALSE), eval)
toc()
tic()
#Average the models with a choosen set of models
SavedAveragedModels <- model.avg(wine11, subset = delta < 100)
toc()
SavedAveragedModels
#####################################################################################
#################### Perform average model.fit on train data ########################
#####################################################################################
wineaveragefittraindata.prob = predict(SavedAveragedModels, winetrainfull, type="response")

ROUNDwineaveragefittraindata.prob <- round(wineaveragefittraindata.prob, digits = 3)
wineaveragetrainpred <- ifelse(ROUNDwineaveragefittraindata.prob >= "0.5", 1, 0)
fwineaveragetrainpred <- factor(wineaveragetrainpred)
levels(fwineaveragetrainpred) = c('Bad Quality','Good Quality')
#Use "fwinetrainreal" from logit example as reference (it is the same)

#Creating confusion matrix as a simple overview
ConfMataveragetrain <- confusionMatrix(data=fwineaveragetrainpred, reference = fwinetrainreal) #Reference = Real Data

#Display results 
ConfMataveragetrain

#####################################################################################
################ Perform average model.fit on test data #############################
#####################################################################################
wineaveragefittestdata.prob = predict(SavedAveragedModels, winetest, type="response")

ROUNDwineaveragefittestdata.prob <- round(wineaveragefittestdata.prob, digits = 3)
wineaveragetestpred <- ifelse(ROUNDwineaveragefittestdata.prob >= "0.5", 1, 0)
fwineaveragetestpred <- factor(wineaveragetestpred)
levels(fwineaveragetestpred) = c('Bad Quality','Good Quality')

#Use "fwinetestreal" from logit example as reference (it is the same)

#Creating confusion matrix
ConfMataveragetest <- confusionMatrix(data=fwineaveragetestpred, reference = fwinetestreal)

#Display results 
ConfMataveragetest

#####################################################################################
# Comparison of AVERAGE Confusion Matrix - AVERAGE Insample VS AVERAGE Out-Of-Sample
#####################################################################################
#Plot Confusion Matrix with function "draw_confusion_matrix_all", execute all if not already done!
#Exact
draw_confusion_matrix_wineaverage_train(ConfMataveragetrain)
draw_confusion_matrix_wineaverage_test(ConfMataveragetest)
#Plot
par(mfrow=c(1,2)) #2 graphs side by side
fourfoldplot(ConfMataveragetrain$table, main = "Average Traindata")
fourfoldplot(ConfMataveragetest$table, main = "Average Testdata")

#####################################################################################
########################## Compare Coefficient tables ###############################
#####################################################################################
library(jtools)
library(kableExtra)
library(ggstance)
library(broom.mixed)
library(huxtable)
library(texreg)

#direct (not possible in the same table)
screenreg(SavedAveragedModels, custom.model.names = c("Full Averaged Model"))
screenreg(winelogittraindata.fit, custom.model.names = c("Logit Model"))#not ordered


#####################################################################################
############ Confusion Matrix - Comparison of Testdata results #####################
#####################################################################################
#Confusion Matrix - Comparison of Traindata results
#Exact
draw_confusion_matrix_wine_train(ConfMattrain)
draw_confusion_matrix_wineaverage_train(ConfMataveragetrain)
#Plot
par(mfrow=c(1,2)) #2 graphs side by side
fourfoldplot(ConfMattrain$table, main = "Logit Traindata")
fourfoldplot(ConfMataveragetrain$table, main = "Average Traindata")

#Confusion Matrix - Comparison of Testdata results
#Exact
draw_confusion_matrix_wine_test(ConfMattest)
draw_confusion_matrix_wineaverage_test(ConfMataveragetest)
#Plot
par(mfrow=c(1,2)) #2 graphs side by side
fourfoldplot(ConfMattest$table, main = "Logit Testdata")
fourfoldplot(ConfMataveragetest$table, main = "Average Testdata")

#####################################################################################
############ ROC of Performance with Testdata #######################################
#####################################################################################
library(ROCR)

############### Score train data which was predicted with logit.fit ##################
#Take the prediction "banklogitfittestdata.prob"
predlogittrain = prediction(winelogitfittraindata.prob, winetrainreal)

#Perform AUC
auclogittrain = performance(predlogittrain,"auc")@y.values[[1]][1]

#plot ROC curve of Logit Testdataprediction
perflogittrain <- performance(predlogittrain,"tpr","fpr")
plot(perflogittrain, colorize=TRUE, cex.main=1,
     main= paste("Logistic Regression Traindata ROC Curve: AUC =", round(auclogittrain,5)))
abline(a=0, b = 1, col='darkorange1')


############### Score train data which was predicted with average.fit ##################
predAtrain = prediction(wineaveragefittraindata.prob, winetrainreal)

#Perform AUC
aucAtrain = performance(predAtrain,"auc")@y.values[[1]][1]

#plot ROC curve of averaged Traindataprediction
perfAtrain <- performance(predAtrain,"tpr","fpr")
plot(perfAtrain, col="blue",colorize=TRUE , cex.main=1,
     main= paste("Model Averaged Regression Traindata ROC Curve: AUC =", round(aucAtrain,5)))
abline(a=0, b = 1, col='darkorange1')


############### Score test data which was predicted with logit.fit ##################
#Take the prediction "winelogitfittestdata.prob"
predlogittest = prediction(winelogitfittestdata.prob, winetestreal)

#Perform AUC
auclogittest = performance(predlogittest,"auc")@y.values[[1]][1]

#plot ROC curve of Logit Testdataprediction
perflogittest <- performance(predlogittest,"tpr","fpr")
plot(perflogittest, colorize=TRUE, cex.main=1,
     main= paste("Logistic Regression Testdata ROC Curve: AUC =", round(auclogittest,5)))
abline(a=0, b = 1, col='darkorange1')


############### Score test data which was predicted with average.fit ##################
predAtest = prediction(wineaveragefittestdata.prob, winetestreal)

#Perform AUC
aucAtest = performance(predAtest,"auc")@y.values[[1]][1]

#plot ROC curve of averaged Testdataprediction
perfAtest <- performance(predAtest,"tpr","fpr")
plot(perfAtest, col="blue",colorize=TRUE , cex.main=1, xlab="False Positive Rate", ylab="True Positive Rate",
     main= paste("Model Averaged Regression Testdata ROC Curve: AUC =", round(aucAtest,5)))
abline(a=0, b = 1, col='darkorange1')

#####################################################################################
############### Distribution of Prediction Differences (Absolute) ###################
#####################################################################################

######Logit Bank Prediction
Logitfit.prob <- data.frame(winelogitfittestdata.prob)
winereal <- data.frame(as.numeric(winetestreal))

LogitProb.diff <- cbind(winereal,Logitfit.prob)
library(tidyverse)
LogitProb.diff <- LogitProb.diff %>% 
  mutate(Logitdiff = winereal-Logitfit.prob)
LogitProb.diff <- LogitProb.diff %>% 
  mutate(absDiff = abs(Logitdiff))
#Averaged Deviation of all values
colMeans(LogitProb.diff$absDiff)

# Look at positive values which indicate a classification of 1 (Good Quality)
#Positiv 1 - Good Quality
positiv <- subset(LogitProb.diff, winetestreal!=0)
colMeans(positiv$Logitdiff)

# Look at negative values which indicate a classification of 0 (Bad Quality)
#Negativ 0 - Bad Quality
negativ <- subset(LogitProb.diff, winetestreal!=1)
colMeans(negativ$Logitdiff)

#check if the results are correct => Yes
colMeans(LogitProb.diff$absDiff)
table(winetestreal)

#######Wine Average Prediction
Logitfit.probA <- data.frame(wineaveragefittestdata.prob)
winereal <- data.frame(as.numeric(winetestreal))

LogitProb.diffA <- cbind(winereal,Logitfit.probA)
library(tidyverse)
LogitProb.diffA <- LogitProb.diffA %>% 
  mutate(LogitdiffA = winereal-Logitfit.probA)
LogitProb.diffA <- LogitProb.diffA %>% 
  mutate(absDiffA = abs(LogitdiffA))
#Averaged Deviation of all values
colMeans(LogitProb.diffA$absDiffA)

# Look at positive values which indicate a classification of 1 (Good Quality)
#Positiv 1 - Good Quality
positivA <- subset(LogitProb.diffA, winetestreal!=0)
colMeans(positivA$LogitdiffA)

# Look at negative values which indicate a classification of 0 (Bad Quality)
#Negativ 0 - Bad Quality
negativA <- subset(LogitProb.diffA, winetestreal!=1)
colMeans(negativA$LogitdiffA)

#check if the results are correct => Yes
colMeans(LogitProb.diffA$absDiffA)
table(winetestreal)

### Copy all these values in a table in Master Thesis ###
Deviations <- as.data.frame(cbind(nrow(negativ), colMeans(negativ$Logitdiff), nrow(positiv), colMeans(positiv$Logitdiff), colMeans(LogitProb.diff$absDiff)))
DeviationsA <- as.data.frame(cbind(nrow(negativA), colMeans(negativA$LogitdiffA), nrow(positivA), colMeans(positivA$LogitdiffA), colMeans(LogitProb.diffA$absDiffA)))
DeviationsTable <- rbind(Deviations, DeviationsA)
colnames(DeviationsTable) <- c("N Negatives","Negative", "N Positives", "Positive", "Arithmetic Mean")
rownames(DeviationsTable) <- c("Logit", "Average")
DeviationsTable

############################## Plot the density of the calculated deviations LOGIT ######
DeviationsPlot <- data.matrix(LogitProb.diff$Logitdiff)
DeviationsPlotA <- data.matrix(LogitProb.diffA$LogitdiffA)

d <- density(DeviationsPlot)
dA <- density(DeviationsPlotA)

par(mfrow=c(1,2)) #2 graphs side by side
h <-hist(DeviationsPlot, breaks=800, xlim = c(-1,1), ylim = c(0,10), xlab="Deviation",
         main="Histogram Deviations - Logit Test Data")
Meanh <- colMeans(LogitProb.diff$absDiff)
lines(c(Meanh,Meanh), c(0,6), col = "green", lwd = 3)#Mean
lines(d, lwd="3", col="red")

#Legend
legend(0, 8, 
       c("Logit Deviation Density","Mean"), 
       lty=c(1,1), 
       lwd=c(4,4),
       col=c("red","green"))


############################## Plot the density of the calculated deviations AVERAGE
hA <-hist(DeviationsPlotA, breaks=800, xlim = c(-1,1), ylim = c(0,10), xlab="Deviation",
          main="Histogram Deviations - Average Test Data")
MeanhA <- colMeans(LogitProb.diffA$absDiffA)
lines(c(MeanhA,MeanhA), c(0,6), col = "green", lwd = 3)#Mean
lines(dA, lwd="2", col="blue")

#Legend
legend(0, 8, 
       c("Average Deviation Density","Mean"), 
       lty=c(1,1), 
       lwd=c(4,4),
       col=c("blue","green"))


############### Density Comparison in same plot ##############################
par(mfrow=c(1,2)) #2 graphs
plot(d, col="red", type = "l", xlim = c(-1,1), ylim = c(0,3.5), main = "Density Comparison of Deviation - Logit", xlab="Deviation" , ylab="Density")
#Legend
legend(0,3.5, 
       c("Logit Deviation Density"), 
       lty=c(1), 
       lwd=c(4),
       col=c("red"))

plot(dA, col="blue", type = "l", xlim = c(-1,1), ylim = c(0,3.5), main = "Density Comparison of Deviation - Average", xlab="Deviation" , ylab="Density")
#Legend
legend(-0.03,3.5, 
       c("Average Deviation Density"), 
       lty=c(1), 
       lwd=c(4),
       col=c("blue"))

#####################################################################################
#################################### The End ########################################
#####################################################################################

