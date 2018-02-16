txt2matrix <- function(text, master, useEmbedding=FALSE) {
  contents <- sapply(text, function(x){strsplit(x,'')})[[1]]
  if(useEmbedding) 
    return(as.integer(factor(contents, levels=master)))
  else
    return(as.integer(master %in% contents))
}

eval_title <- function(model, title, categorymaster, titleschrmaster,maxtitlelength,useEmbedding=FALSE, threshould=0.5) {
  n <- maxtitlelength - nchar(title)
  et <- txt2matrix(paste(c(title, rep(' ',n)), collapse=''), titleschrmaster,useEmbedding)
  et <- matrix(et, nrow=1)
  if(!is.na(threshould))
    categorymaster[which(model %>% predict(et) > threshould)]
  else {
    r <- as.vector(model %>% predict(et))
    names(r) <- categorymaster
    data.frame(score=sort(r,decreasing=TRUE))
  }
}

eval_multilabel <- function(model, eval_data, threshould=0.5, method=c("identical", "xor"), allowance=0.1) {
  method <- match.arg(method)
  prediction <- 1*(predict(model, eval_data$X) > threshould)
  eval <- sapply(1:dim(prediction)[1], function(i) {
    if(method=="identical")
      identical(as.integer(prediction[i,]), eval_data$Y[i,])
    else if(method=="xor") # return TRUE only when proportion of disagreement between prediction and eval data is lower than allowance.
      sum(xor(as.integer(prediction[i,]),eval_data$Y[i,]))/length(eval_data$Y[i,]) < allowance
  })
  sum(eval) / length(eval)
}