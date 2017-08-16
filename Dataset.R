library(readr)
ctr_full <- read_csv("ctr_data_j.csv")
pickup <- sample(1:nrow(ctr_full),15000)
ctr_sample_train <- ctr_full[pickup,]
pickup <- sample( (1:nrow(ctr_full))[-pickup],10000)
ctr_sample_eval <- ctr_full[pickup,]

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

generate_onehot_masters <- function(fulldata) {
  researchTitles <- as.data.frame(fulldata[,"試験名/Official scientific title of the study"], stringsAsFactors = FALSE)[,1]
  diseaseClasses <- as.data.frame(fulldata[,"疾患区分1/Classification by specialty"],stringsAsFactors = FALSE)[,1]

    nadata <- is.na(researchTitles) | is.na(diseaseClasses)
  researchTitles <- researchTitles[!nadata]
  diseaseClasses <- diseaseClasses[!nadata]

  mc <- strsplit(diseaseClasses, ':')
  categoryMaster <- levels(factor(unlist(mc)))
  
  max_title_length <- max(sapply(researchTitles, nchar))
  tc <- strsplit(researchTitles, '')
  titlesCharactersMaster <- levels(factor(c(' ', unlist(tc))))
  return(list(categoryMaster=categoryMaster, titlesCharactersMaster=titlesCharactersMaster, max_title_length=max_title_length))
}

generate_xy <- function(data, categoryMaster, titlesCharactersMaster, max_title_length, forRNN=FALSE) {
  researchTitles <- as.data.frame(data[,"試験名/Official scientific title of the study"], stringsAsFactors = FALSE)[,1]
  diseaseClasses <- as.data.frame(data[,"疾患区分1/Classification by specialty"],stringsAsFactors = FALSE)[,1]
  
  nadata <- is.na(researchTitles) | is.na(diseaseClasses)
  researchTitles <- researchTitles[!nadata]
  diseaseClasses <- diseaseClasses[!nadata]
  
  ## Prepare Y data(classification)
  mc <- strsplit(diseaseClasses, ':')
  mcc <- sapply(mc, function(x) {
      as.integer(categoryMaster %in% x)
  }, simplify="array")
  diseaseClasses_onehot <- t(mcc)
  Y <- diseaseClasses_onehot
  
  ## Prepare X data (one-hot / 1-of-N coded text)
  
  tc <- strsplit(researchTitles, '')
  titles_padded <- sapply(researchTitles, function(x) {
    n <- max_title_length - nchar(x)
    paste(c(x, rep(' ',n)), collapse='')
  })
  
  tcc <- sapply(titles_padded,txt2matrix, titlesCharactersMaster, forRNN,  simplify="array")
  if(forRNN)
    researchTitles_onehot <- aperm(tcc, perm=c(3,2,1))
  else
    researchTitles_onehot <- aperm(tcc, perm=c(2,1))
  
  X <- researchTitles_onehot

  return(list(X=X,Y=Y))
}

ohmasters <- generate_onehot_masters(ctr_full)

train_data <- generate_xy(ctr_sample_train, ohmasters$categoryMaster, ohmasters$titlesCharactersMaster, ohmasters$max_title_length)
save(train_data, ohmasters, file="Train15000.RData")
eval_data <- generate_xy(ctr_sample_eval, ohmasters$categoryMaster, ohmasters$titlesCharactersMaster, ohmasters$max_title_length)
save(eval_data, ohmasters, file="Eval10000.RData")
