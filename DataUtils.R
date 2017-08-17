txt2matrix <- function(text, master, forRNN) {
  contents <- sapply(text, function(x){strsplit(x,'')})[[1]]
  if(forRNN)
    return(sapply(contents, function(x) {
      as.integer(master %in% x)
    }))
  else
    return(as.integer(master %in% contents))
}

eval_title <- function(model, title, categorymaster, titleschrmaster,maxtitlelength,forRNN=FALSE, threshould=0.5) {
  n <- maxtitlelength - nchar(title)
  et <- txt2matrix(paste(c(title, rep(' ',n)), collapse=''), titleschrmaster,forRNN)
  if(forRNN)
    et <- aperm(et, perm=c(3,2,1))
  else
    et <- matrix(et, nrow=1)
  
  categorymaster[which(model %>% predict(et) > threshould)]
}
