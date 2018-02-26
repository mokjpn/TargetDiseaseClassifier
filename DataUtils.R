eval_multilabel <- function(model, eval_data, threshould=0.5, rankmax=5, method=c("identical", "rank", "xor", "maxidentical"), allowance=0.1) {
  method <- match.arg(method)
  prediction <- predict(model, eval_data$X)
  eval <- sapply(1:dim(prediction)[1], 
                 function(i) {
                   if(method=="identical")
                     identical(1.0*(prediction[i,] > threshould),eval_data$Y[i,])
                   else if(method=="xor") # return TRUE only when proportion of disagreement between prediction and eval data is lower than allowance.
                     sum(xor(1.0*(prediction[i,] > threshould),eval_data$Y[i,]))/length(eval_data$Y[i,]) < allowance
                   else if(method=="rank") # return TRUE only when top (rankmax)th of predicted probability includes eval data.
                     all(which(eval_data$Y[i,]==1) %in% (order(prediction[i,],decreasing=TRUE)[1:rankmax])) # all->anyとすると、categorical_accuracyと同値
                   else if(method=="maxidentical")
                     identical(which.max(prediction[i,]),which.max(eval_data$Y[i,]))
                 })
  sum(eval) / length(eval)
}

