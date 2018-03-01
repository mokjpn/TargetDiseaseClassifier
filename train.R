library(keras)
source("DataUtils.R")
source("doTrain.R")
#use_backend("plaidml")

set.seed(38934)

source=c("title", "purpose")
tokenize=c("char", "word")
embed=c("ohm", "seq")
target=c("class", "type")
network=c("dense", "rnn", "1dc")

build1 <- expand.grid(source,tokenize,embed,target,network, stringsAsFactors = FALSE)
embed="bow"
network="dense"
build2 <- expand.grid(source,tokenize,embed,target,network, stringsAsFactors = FALSE)
build <- rbind(build1, build2)

build$loss <- NA
build$acc <- NA
build$identical <- NA
build$rank <- NA
for(i in 1:nrow(build)) {
  res <- doTrain(build[i,1], build[i,2], build[i,3], build[i,4], build[i,5], large=FALSE)
  build$loss[i] <- res$loss_and_metrics$loss
  build$acc[i] <- res$loss_and_metrics$categorical_accuracy
  build$identical[i] <- res$eval_multilabel_identical
  build$rank[i] <- res$eval_multilabel_rank
}

write.csv(build, file="build.csv")

#stop()
source=c("title", "purpose")
tokenize=c("char", "word")
embed="seq"
target=c("class", "type")
network=c("dense", "rnn", "1dc")

build1 <- expand.grid(source,tokenize,embed,target,network, stringsAsFactors = FALSE)
embed="bow"
network="dense"
build2 <- expand.grid(source,tokenize,embed,target,network, stringsAsFactors = FALSE)
build <- rbind(build1, build2)

build$loss <- NA
build$acc <- NA
build$identical <- NA
build$rank <- NA
for(i in 1:nrow(build)) {
  res <- doTrain(build[i,1], build[i,2], build[i,3], build[i,4], build[i,5], large=TRUE)
  build$loss[i] <- res$loss_and_metrics$loss
  build$acc[i] <- res$loss_and_metrics$categorical_accuracy
  build$identical[i] <- res$eval_multilabel_identical
  build$rank[i] <- res$eval_multilabel_rank
}

write.csv(build, file="build-large.csv")


