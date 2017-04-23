# Create Submission ####
dsub=read.csv("submission.csv", header = T, sep = ",")
dsub$Label=finpred
write.csv(dsub, "submission.csv",row.names = F)

# load libraries ####
library(data.table)
library(caret)
library(car)
library(Boruta)
library(mxnet)
library(h2o)
source("D:/R_Data/functions.R")

setwd("D:/Datasets/Digit Recognizer")

# load data files ####
train.raw=fread("train.csv", stringsAsFactors = T)
test.raw=fread("test.csv", stringsAsFactors = T)
summary(train.raw)
summary(test.raw)
dtrain=as.data.frame(train.raw)
dtest=as.data.frame(test.raw)

# convert label to factor
resp=(train.raw$label)
dtrain=dtrain[,-1]

# convert all columns to numeric
dtrain=as.data.frame(sapply(dtrain, as.numeric))
dtest=as.data.frame(sapply(dtest, as.numeric))

# remove variables with zero variance ####
col.keep=zeroVar(dtrain)
dtrain=dtrain[,col.keep]

# Run Boruta to select predictors
bor.trn=Boruta(x = dtrain, y = resp, maxRuns = 81, doTrace = 2)
conf.attr=getSelectedAttributes(bor.trn, withTentative = F)
dstrain=dtrain[,names(dtrain) %in% conf.attr]
dstest=dtest[,names(dtest) %in% conf.attr]

# normalize predictors
dtrain2=dtrain
dtest2=dtest
dtrain=dtrain/255
dtrain=as.data.frame(dtrain)
dtest=dtest/255
dtest=as.data.frame(dtest)

# separate digits
resp.d=as.data.frame(resp)
digits=as.data.frame(model.matrix(~., data = resp.d))
digits=digits[,-1]
digits$resp0=apply(digits, 1, function(x)(ifelse(sum(x)==0,1,0)))

# split train and validation set ####
set.seed(713)
prt=createDataPartition(resp, p=0.7, list = F)
dt=dtrain[prt,]
dv=dtrain[-prt,]
resp.t=resp[prt]
resp.v=resp[-prt]



# Create random forest model ####
set.seed(1)
cvCtrl=trainControl(method = "repeatedcv", repeats = 2, number=2, verboseIter = T)
rfmod=train(x=dt, y=as.factor(resp.t), method="rf", trControl = cvCtrl, tuneLength = 2)
# build final predictions
rfmod.f=predict(rfmod, newdata = dtest)

# Create extreme gradient boost model ####
# Linear XGB
set.seed(713)
xgGrid = expand.grid(nrounds = c(100,150,200), lambda = 1, alpha = 0, eta = 0.1)
cvCtrl=trainControl(method = "repeatedcv", repeats = 1, number=2, verboseIter = T)
xgmod=train(x=dt, y=resp.t, method="xgbLinear", trControl = cvCtrl, tuneGrid = xgGrid)
xgmod.p=predict(xgmod, newdata = dv)
xgmod.f=predict(xgmod, newdata = dt)
table(resp.v, xgmod.p)
table(resp.t, xgmod.f)
sum(diag(table(resp.v, xgmod.p)))/ length(xgmod.p)
xgmod.s=predict(xgmod, newdata = dtest)

# Tree XGB
set.seed(713)
xgTgrid=expand.grid(nrounds = c(100,150), eta = c(0.1,0.2), max_depth = c(4,6), gamma = c(0.1,0.2),
                    min_child_weight = c(4, 5), colsample_bytree = c(0.5, 0.9))
cvCtrl=trainControl(method = "repeatedcv", repeats = 1, number=2, verboseIter = T)
xgTmod=train(x=dt, y=resp.t, method="xgbTree", trControl = cvCtrl, tuneLength = 2)
xgTmod.p=predict(xgTmod, newdata = dv)
xgTmod.f=predict(xgTmod, newdata = dt)
table(resp.v, xgmod.p)
sum(diag(table(resp.v, xgmod.p)))/ length(xgmod.p)
xgTmod.s=predict(xgTmod, newdata = dtest)
das=data.frame(all.pred$XGBT, xgTmod.s)
# Create Neural Net models ####
# Digit 1
set.seed(713)
cvCtrl=trainControl(method = "repeatedcv", repeats = 1, number=2, verboseIter = T)
nnmod1=train(x=dt, y=digits.t$resp1, method="nnet", trControl = cvCtrl, tuneLength = 3)
nnmod2=train(x=dt, y=digits.t$resp2, method="nnet", trControl = cvCtrl, tuneLength = 3)
nnmod3=train(x=dt, y=digits.t$resp3, method="nnet", trControl = cvCtrl, tuneLength = 3)
nnmod4=train(x=dt, y=digits.t$resp4, method="nnet", trControl = cvCtrl, tuneLength = 3)
nnmod5=train(x=dt, y=digits.t$resp5, method="nnet", trControl = cvCtrl, tuneLength = 3)
nnmod6=train(x=dt, y=digits.t$resp6, method="nnet", trControl = cvCtrl, tuneLength = 3)
nnmod7=train(x=dt, y=digits.t$resp7, method="nnet", trControl = cvCtrl, tuneLength = 3)
nnmod8=train(x=dt, y=digits.t$resp8, method="nnet", trControl = cvCtrl, tuneLength = 3)
nnmod9=train(x=dt, y=digits.t$resp9, method="nnet", trControl = cvCtrl, tuneLength = 3)
nnmod0=train(x=dt, y=digits.t$resp0, method="nnet", trControl = cvCtrl, tuneLength = 3)
dv.p=as.data.frame(matrix(nrow = nrow(dv), ncol = 10))
names(dv.p)=names(digits)
dv.p$resp1=predict(nnmod1, newdata = dv)
dv.p$resp2=predict(nnmod2, newdata = dv)
dv.p$resp3=predict(nnmod3, newdata = dv)
dv.p$resp4=predict(nnmod4, newdata = dv)
dv.p$resp5=predict(nnmod5, newdata = dv)
dv.p$resp6=predict(nnmod6, newdata = dv)
dv.p$resp7=predict(nnmod7, newdata = dv)
dv.p$resp8=predict(nnmod8, newdata = dv)
dv.p$resp9=predict(nnmod9, newdata = dv)
dv.p$resp0=predict(nnmod0, newdata = dv)
dv.p$label=max.col(dv.p[,c(1:10)])
dv.p$label=recode(dv.p$label, "10=0")
table(resp.v, dv.p$label)

# Build SVM model (BEST) ####
set.seed(713)
svmGrid=expand.grid(degree = c(2,3), scale = 0.001, C = 0.35)
cvCtrl=trainControl(method = "repeatedcv", repeats = 1, number=2, verboseIter = T)
svmmod=train(x=dt, y=resp.t, method="svmPoly", trControl = cvCtrl, tuneGrid = svmGrid)
svmmod.p=predict(svmmod, newdata = dv)
sum(diag(table(resp.v, svmmod.p)))/ length(svmmod.p)
svmmod.f=predict(svmmod, newdata = dtest)

set.seed(713)
svmGrid=expand.grid(degree = c(2,3), scale = 0.001, C = 0.35)
cvCtrl=trainControl(method = "repeatedcv", repeats = 1, number=2, verboseIter = T)
svmRmod=train(x=dt, y=resp.t, method="svmRadial", trControl = cvCtrl, tuneLength = 2)
svmRmod.p=predict(svmRmod, newdata = dv)
sum(diag(table(resp.v, svmRmod.p)))/ length(svmRmod.p)
svmRmod.f=predict(svmRmod, newdata = dtest)

# Build GBM model ####
set.seed(713)
cvCtrl=trainControl(method = "repeatedcv", repeats = 1, number=2, verboseIter = T)
gbmmod=train(x=dt, y=resp.t, method="gbm", trControl = cvCtrl, tuneLength = 2)
gbmmod.p=predict(gbmmod, newdata = dv)
sum(diag(table(resp.v, gbmmod.p)))/ length(gbmmod.p)
gbmmod.f=predict(gbmmod, newdata = dtest)

# Fit CNN ####
dtraint=t(dtrain)
dtestt=t(dtest)
# Transpose
dtt=t(dt)
dvt=t(dv)

# Build Convventional NN ####
data1 <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data1, name="fc1", num_hidden=128)
act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=64)
act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="relu")
fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=10)
softmax <- mx.symbol.SoftmaxOutput(fc3, name="sm")

mx.set.seed(713)
snnmod <- mx.model.FeedForward.create(softmax, X=dtt.arr, y=resp.t,
                                     ctx=devcs, num.round=25, array.batch.size=100,
                                     learning.rate=0.05, momentum=0.8,  eval.metric=mx.metric.accuracy,
                                     initializer=mx.init.uniform(0.07),
                                     epoch.end.callback=mx.callback.log.train.metric(100),
                                     batch.end.callback = mx.callback.log.train.metric(100))

snnmod.c=predict(snnmod, dtt.arr)
snnmod.c.b=apply(snnmod.c, 2, function(x) which.max(x))
sum(diag(table(resp.t,snnmod.c.b)))/ length(snnmod.c.b)
snnmod.d=predict(snnmod, dvt.arr)
snnmod.d.b=apply(snnmod.d, 2, function(x) which.max(x))
sum(diag(table(resp.v,snnmod.d.b)))/ length(snnmod.d.b)
snnmod.p=predict(snnmod, dtestt.arr)
snnmod.p.b=apply(snnmod.p, 2, function(x) which.max(x))-1

# Build network LeNet ####
# input
data <- mx.symbol.Variable('data')
# first conv
conv1 <- mx.symbol.Convolution(data=data, kernel=c(5,5), num_filter=20)
tanh1 <- mx.symbol.Activation(data=conv1, act_type="tanh")
pool1 <- mx.symbol.Pooling(data=tanh1, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))
# second conv
conv2 <- mx.symbol.Convolution(data=pool1, kernel=c(5,5), num_filter=50)
tanh2 <- mx.symbol.Activation(data=conv2, act_type="tanh")
pool2 <- mx.symbol.Pooling(data=tanh2, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))
# first fullc
flatten <- mx.symbol.Flatten(data=pool2)
fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 <- mx.symbol.Activation(data=fc1, act_type="tanh")
# second fullc
fc2 <- mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
# loss
lenet <- mx.symbol.SoftmaxOutput(data=fc2)

# Reshape into matrices
# Convert to array
dtt.arr=dtt
dim(dtt.arr)=c(28,28,1, ncol(dtt))
dvt.arr=dvt
dim(dvt.arr)=c(28,28,1, ncol(dvt))
dtestt.arr=dtestt
dim(dtestt.arr)=c(28,28,1, ncol(dtestt.arr))

# set device
devcs=mx.cpu()

# Fit model
mx.set.seed(666)
tic <- proc.time()
lenetmod <- mx.model.FeedForward.create(lenet, X=dtt.arr, y=resp.t,
                                     ctx=devcs, num.round=15, array.batch.size=100,
                                     learning.rate=0.05, momentum=0.9, wd=0.00001,
                                     eval.metric=mx.metric.accuracy,
                                     epoch.end.callback=mx.callback.log.train.metric(100),
                                     batch.end.callback = mx.callback.log.train.metric(50))

print(proc.time() - tic)
lenetmod.c=predict(lenetmod, dtt.arr)
lenetmod.c.b=apply(lenetmod.c, 2, function(x) which.max(x))
sum(diag(table(resp.t,lenetmod.c.b)))/ length(lenetmod.c.b)
lenetmod.d=predict(lenetmod, dvt.arr)
lenetmod.d.b=apply(lenetmod.d, 2, function(x) which.max(x))
sum(diag(table(resp.v,lenetmod.d.b)))/ length(lenetmod.d.b)
lenetmod.p=predict(lenetmod, dtestt.arr)
lenetmod.p.b=apply(lenetmod.p, 2, function(x) which.max(x))-1

# H2O ####
localH2O = h2o.init(ip = "localhost", port = 54321, startH2O = TRUE, 
                    max_mem_size = '12g', nthreads = -1)

dth=dt
dth$labl=resp.t
h2o.dt = as.h2o(dth)
h2o.dv = as.h2o(dv)
h2o.test =  as.h2o(dtest)

set.seed(1723)
tic=proc.time()
h2omod=h2o.deeplearning(x = 1:(ncol(dth)-1),
                        y = "labl",
                        training_frame = h2o.dt,
                        activation = "TanhWithDropout",
                        input_dropout_ratio = 0.2,
                        hidden_dropout_ratios = c(0.4,0.4),
                        balance_classes = T, 
                        hidden = c(500,400),
                        momentum_stable = 0.99,
                        nesterov_accelerated_gradient = T,
                        epochs = 10,
                        #loss = "CrossEntropy",
                        #categorical_encoding = "OneHotInternal"
)

h2o.confusionMatrix(h2omod)
h2omod.d=as.data.frame(h2o.predict(h2omod, h2o.dt, type="raw"))
sum(diag(table(resp.t, h2omod.d[,1])))/ length(resp.t)
h2omod.c=as.data.frame(h2o.predict(h2omod, h2o.dv, type = "raw"))
sum(diag(table(resp.v, h2omod.c[,1])))/ length(resp.v)
h2omod.a=as.data.frame(h2o.predict(h2omod, h2o.test, type="raw"))
h2omod.p=h2omod.a[,1]
h2o.shutdown()

# Ensemble model ####
all.pred=as.data.frame(matrix(nrow = nrow(dtest)))
all.pred.dv=as.data.frame(matrix(nrow = nrow(dv)))
all.pred$LeNet=lenetmod.p.b
all.pred.dv$LeNet=lenetmod.d.b
all.pred$SNN=snnmod.p.b
all.pred.dv$SNN=snnmod.d.b
all.pred$XGBL=xgmod.s
all.pred$XGBT=xgTmod.s
all.pred$SVMP=svmmod.f
all.pred$SVMR=svmRmod.f
all.pred$GBM=gbmmod.f
all.pred$H2O=h2omod.p
finpred=apply(all.pred[,-9],1, function(x) Mode(x))


Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}



write.csv(all.pred, "AllPred.csv", row.names = F)
write.csv(all.pred.dv, "AllPredDv", row.names = F)

all.pred=read.csv("AllPred.csv", header = T, sep = ",")
all.pred.dv=read.csv("AllPredDv", header = T, sep = ",")
