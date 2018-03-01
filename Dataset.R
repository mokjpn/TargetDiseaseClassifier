library(readr)
library(keras)
library(RMeCab)
source("DataUtils.R") # to_bow(), generate_charmasters(), generate_wordmasters()
#use_backend("plaidml")

set.seed(38934)
WITH_ONEHOT = TRUE

if(WITH_ONEHOT) {
  NSAMPLES=300
  DATADIR="data"
} else {
  NSAMPLES=10000
  DATADIR="data_large"
}

ctr_raw <- as.data.frame(read_csv("ctr_data_j.csv"))

# One-Hot Vector作成時にメインメモリが不足するため、性能評価では1/30のデータを用いる
if(WITH_ONEHOT)
  ctr_raw <- ctr_raw[sample(1:nrow(ctr_raw), nrow(ctr_raw)/30),]

title_col <- "試験名/Official scientific title of the study"
purpose_col <- "目的1/Narrative objectives1"
classes_col <- "疾患区分1/Classification by specialty"
types_col <- "試験の種類/Study type"

## 利用する全項目に欠損がないものに限定
ctr_full <- ctr_raw[!(is.na(ctr_raw[,title_col]) | is.na(ctr_raw[,purpose_col]) | is.na(ctr_raw[,classes_col]) | is.na(ctr_raw[,types_col])),]

## 複数選択の疾患区分を分解して、カテゴリーリストを作成
DiseaseClassesMaster  <- levels(factor(unlist(strsplit(ctr_full[,classes_col], ":"))))
## 試験の種類のカテゴリーリストを作成
TypesMaster <- levels(factor(ctr_full[,types_col]))
save(DiseaseClassesMaster, TypesMaster, file=paste(DATADIR,"/","PredictionMasters.RData",sep=""))

## カテゴリーリストを用いて、疾患区分および試験の種類を整数の番号に変換
ctr_full_classseq <- sapply(ctr_full[,classes_col], function(s) which(DiseaseClassesMaster %in% unlist(strsplit(s, ":")) ),USE.NAMES=FALSE)
ctr_full_typeseq <- sapply(ctr_full[,types_col], function(s) which(TypesMaster == s),USE.NAMES=FALSE)

cf_bow_class <- t(sapply(ctr_full_classseq, function(s) unlist(to_bow(s, length(DiseaseClassesMaster)))))
cf_bow_type <- t(sapply(ctr_full_typeseq, function(s) unlist(to_bow(s, length(TypesMaster)))))

# 自然言語文を文字単位で整数にエンコーディング

## 研究タイトルと研究目的の文字一覧を作成
title_chars <- generate_charmasters(ctr_full[,title_col])
purpose_chars <- generate_charmasters(ctr_full[,purpose_col])

## 文字単位に分解した研究タイトルを数値化できるtokenizerを作成して、研究タイトルを整数の文字番号の集合に変換
cf_title_char <- sapply(ctr_full[,title_col], function(s){strsplit(s,'')[[1]]})
tokenizer_title <- text_tokenizer(filters="") %>% fit_text_tokenizer(title_chars$index)
ctr_full_titleseq <- sapply(cf_title_char, function(s) unlist(texts_to_sequences(tokenizer_title,s)))
ctr_full_titleseq_mat <- pad_sequences(unname(ctr_full_titleseq), title_chars$MaxLength) # unnameはkerasのバグ回避
save_text_tokenizer(tokenizer_title, paste(DATADIR, "/", "title_char_tokenizer.keras", sep=""))

## 文字単位に分解した研究目的を数値化できるtokenizerを作成して、研究目的を整数の文字番号の集合に変換
cf_purpose_char <- sapply(ctr_full[,purpose_col], function(s){strsplit(s,'')[[1]]})
tokenizer_purpose <- text_tokenizer(filters="") %>% fit_text_tokenizer(purpose_chars$index)
ctr_full_purposeseq <- sapply(cf_purpose_char, function(s) unlist(texts_to_sequences(tokenizer_purpose,s)))
ctr_full_purposeseq_mat <- pad_sequences(unname(ctr_full_purposeseq), purpose_chars$MaxLength) # unnameはkerasのバグ回避
save_text_tokenizer(tokenizer_purpose, paste(DATADIR, "/", "purpose_char_tokenizer.keras",sep=""))

# 自然言語文をRMeCab+万病辞書で単語単位で整数にエンコーディング

## 研究タイトルと研究目的の単語一覧を作成
MINIMUM_WORD_FREQUENCY <- 3 # 全データを通して、最低3回以上出現した単語のみをインデックス
title_words <- generate_wordmasters(ctr_full[,title_col], MINIMUM_WORD_FREQUENCY)
purpose_words <- generate_wordmasters(ctr_full[,purpose_col], MINIMUM_WORD_FREQUENCY)

## 単語単位に分解した研究タイトルを数値化できるtokenizerを作成して、研究タイトルを整数の単語番号の集合に変換
cf_title_word <- sapply(ctr_full[,title_col], function(s) unlist(RMeCabC(s)))
tokenizer_title_words <- text_tokenizer(filters="") %>% fit_text_tokenizer(title_words$index)
ctr_full_title_wordsseq <- sapply(cf_title_word, function(s) unlist(texts_to_sequences(tokenizer_title_words,s)))
ctr_full_title_wordsseq[sapply(ctr_full_title_wordsseq,length)==0] <- tokenizer_title_words$word_index[1] # 稀な単語だけで構成される文章がNULLになってしまう場合があるので，その時はtokenizerのword indexの最初の要素を単独で持つものとする
ctr_full_title_wordsseq_mat <- pad_sequences(unname(ctr_full_title_wordsseq), title_words$MaxLength) # unnameはkerasのバグ回避
save_text_tokenizer(tokenizer_title_words, paste(DATADIR, "/", "title_word_tokenizer.keras",sep=""))

## 単語単位に分解した研究目的を数値化できるtokenizerを作成して、研究目的を整数の単語番号の集合に変換
cf_purpose_word <- sapply(ctr_full[,purpose_col], function(s) unlist(RMeCabC(s)))
tokenizer_purpose_words <- text_tokenizer(filters="") %>% fit_text_tokenizer(purpose_words$index)
ctr_full_purpose_wordsseq <- sapply(cf_purpose_word, function(s) unlist(texts_to_sequences(tokenizer_purpose_words,s)))
ctr_full_purpose_wordsseq[sapply(ctr_full_purpose_wordsseq,length)==0] <- tokenizer_purpose_words$word_index[1] # 稀な単語だけで構成される文章がNULLになってしまう場合があるので，その時はtokenizerのword indexの最初の要素を単独で持つものとする
ctr_full_purpose_wordsseq_mat <- pad_sequences(unname(ctr_full_purpose_wordsseq), purpose_words$MaxLength) # unnameはkerasのバグ回避
save_text_tokenizer(tokenizer_purpose_words, paste(DATADIR,"/","purpose_word_tokenizer.keras",sep=""))

# Bag of Words vector に変換: 1文章1ベクトル。最終的な次元は、c(例数,文字種数)

cf_bow_title_char <- t(sapply(ctr_full_titleseq, function(s) unlist(to_bow(s, length(title_chars$index)))))
cf_bow_purpose_char <- t(sapply(ctr_full_purposeseq, function(s) unlist(to_bow(s, length(purpose_chars$index)))))
cf_bow_title_word <- t(sapply(ctr_full_title_wordsseq, function(s) unlist(to_bow(s, length(title_words$index)))))
cf_bow_purpose_word <- t(sapply(ctr_full_purpose_wordsseq, function(s) unlist(to_bow(s, length(purpose_words$index)))))

if(WITH_ONEHOT) {
  # One-Hot Matrixに変換: 1文章1行列。最終的な次元は、c(例数, １例の最大文字数, 文字種数)
  a <- sapply(ctr_full_titleseq_mat, function(s) sequences_to_matrix(tokenizer_title, as.list(s), mode="binary"))
  d <- dim(a)
  dim(a) <- c(d[1],nrow(ctr_full_titleseq_mat), d[2]/nrow(ctr_full_titleseq_mat))
  cf_ohm_title_char <- aperm(a, c(2,3,1))
  
  a <- sapply(ctr_full_purposeseq_mat, function(s) sequences_to_matrix(tokenizer_purpose, as.list(s), mode="binary"))
  d <- dim(a)
  dim(a) <- c(d[1],nrow(ctr_full_purposeseq_mat), d[2]/nrow(ctr_full_purposeseq_mat))
  cf_ohm_purpose_char <- aperm(a, c(2,3,1))
  
  a <- sapply(ctr_full_title_wordsseq_mat, function(s) sequences_to_matrix(tokenizer_title_words, as.list(s), mode="binary"))
  d <- dim(a)
  dim(a) <- c(d[1],nrow(ctr_full_title_wordsseq_mat), d[2]/nrow(ctr_full_title_wordsseq_mat))
  cf_ohm_title_words <- aperm(a, c(2,3,1))
  
  a <- sapply(ctr_full_purpose_wordsseq_mat, function(s) sequences_to_matrix(tokenizer_purpose_words, as.list(s), mode="binary"))
  d <- dim(a)
  dim(a) <- c(d[1],nrow(ctr_full_purpose_wordsseq_mat), d[2]/nrow(ctr_full_purpose_wordsseq_mat))
  cf_ohm_purpose_words <- aperm(a, c(2,3,1))
}

# Embedding Layer
# ctr_full_titleseq_mat等をそのまま投入できるはず。この次元はc(例数, 1例の最大文字数)

pickup_train <- sample(1:nrow(ctr_full),NSAMPLES)
ctr_train <- ctr_full[pickup_train,]
pickup_eval <- sample( (1:nrow(ctr_full))[-pickup_train],NSAMPLES)
ctr_eval <- ctr_full[pickup_eval,]

d_bow_title_char_class <- list(train=list(X=cf_bow_title_char[pickup_train,],Y=cf_bow_class[pickup_train,]), 
                               eval=list(X=cf_bow_title_char[pickup_eval,],Y=cf_bow_class[pickup_eval,]))
d_bow_title_char_type <- list(train=list(X=cf_bow_title_char[pickup_train,],Y=cf_bow_type[pickup_train,]), 
                               eval=list(X=cf_bow_title_char[pickup_eval,],Y=cf_bow_type[pickup_eval,]))
d_bow_title_word_class <- list(train=list(X=cf_bow_title_word[pickup_train,],Y=cf_bow_class[pickup_train,]), 
                               eval=list(X=cf_bow_title_word[pickup_eval,],Y=cf_bow_class[pickup_eval,]))
d_bow_title_word_type <- list(train=list(X=cf_bow_title_word[pickup_train,],Y=cf_bow_type[pickup_train,]), 
                              eval=list(X=cf_bow_title_word[pickup_eval,],Y=cf_bow_type[pickup_eval,]))
d_bow_purpose_char_class <- list(train=list(X=cf_bow_purpose_char[pickup_train,],Y=cf_bow_class[pickup_train,]), 
                               eval=list(X=cf_bow_purpose_char[pickup_eval,],Y=cf_bow_class[pickup_eval,]))
d_bow_purpose_char_type <- list(train=list(X=cf_bow_purpose_char[pickup_train,],Y=cf_bow_type[pickup_train,]), 
                              eval=list(X=cf_bow_purpose_char[pickup_eval,],Y=cf_bow_type[pickup_eval,]))
d_bow_purpose_word_class <- list(train=list(X=cf_bow_purpose_word[pickup_train,],Y=cf_bow_class[pickup_train,]), 
                               eval=list(X=cf_bow_purpose_word[pickup_eval,],Y=cf_bow_class[pickup_eval,]))
d_bow_purpose_word_type <- list(train=list(X=cf_bow_purpose_word[pickup_train,],Y=cf_bow_type[pickup_train,]), 
                              eval=list(X=cf_bow_purpose_word[pickup_eval,],Y=cf_bow_type[pickup_eval,]))
if(WITH_ONEHOT) {
  d_ohm_title_char_class <- list(train=list(X=cf_ohm_title_char[pickup_train,,],Y=cf_bow_class[pickup_train,]), 
                                 eval=list(X=cf_ohm_title_char[pickup_eval,,],Y=cf_bow_class[pickup_eval,]))
  d_ohm_title_char_type <- list(train=list(X=cf_ohm_title_char[pickup_train,,],Y=cf_bow_type[pickup_train,]), 
                                eval=list(X=cf_ohm_title_char[pickup_eval,,],Y=cf_bow_type[pickup_eval,]))
  d_ohm_title_word_class <- list(train=list(X=cf_ohm_title_words[pickup_train,,],Y=cf_bow_class[pickup_train,]), 
                                 eval=list(X=cf_ohm_title_words[pickup_eval,,],Y=cf_bow_class[pickup_eval,]))
  d_ohm_title_word_type <- list(train=list(X=cf_ohm_title_words[pickup_train,,],Y=cf_bow_type[pickup_train,]), 
                                eval=list(X=cf_ohm_title_words[pickup_eval,,],Y=cf_bow_type[pickup_eval,]))
  d_ohm_purpose_char_class <- list(train=list(X=cf_ohm_purpose_char[pickup_train,,],Y=cf_bow_class[pickup_train,]), 
                                   eval=list(X=cf_ohm_purpose_char[pickup_eval,,],Y=cf_bow_class[pickup_eval,]))
  d_ohm_purpose_char_type <- list(train=list(X=cf_ohm_purpose_char[pickup_train,,],Y=cf_bow_type[pickup_train,]), 
                                eval=list(X=cf_ohm_purpose_char[pickup_eval,,],Y=cf_bow_type[pickup_eval,]))
  d_ohm_purpose_word_class <- list(train=list(X=cf_ohm_purpose_words[pickup_train,,],Y=cf_bow_class[pickup_train,]), 
                                   eval=list(X=cf_ohm_purpose_words[pickup_eval,,],Y=cf_bow_class[pickup_eval,]))
  d_ohm_purpose_word_type <- list(train=list(X=cf_ohm_purpose_words[pickup_train,,],Y=cf_bow_type[pickup_train,]), 
                                eval=list(X=cf_ohm_purpose_words[pickup_eval,,],Y=cf_bow_type[pickup_eval,]))
} 
d_seq_title_char_class <- list(train=list(X=ctr_full_titleseq_mat[pickup_train,],Y=cf_bow_class[pickup_train,]), 
                               eval=list(X=ctr_full_titleseq_mat[pickup_eval,],Y=cf_bow_class[pickup_eval,]))
d_seq_title_char_type <- list(train=list(X=ctr_full_titleseq_mat[pickup_train,],Y=cf_bow_type[pickup_train,]), 
                               eval=list(X=ctr_full_titleseq_mat[pickup_eval,],Y=cf_bow_type[pickup_eval,]))
d_seq_title_word_class <- list(train=list(X=ctr_full_title_wordsseq_mat[pickup_train,],Y=cf_bow_class[pickup_train,]), 
                               eval=list(X=ctr_full_title_wordsseq_mat[pickup_eval,],Y=cf_bow_class[pickup_eval,]))
d_seq_title_word_type <- list(train=list(X=ctr_full_title_wordsseq_mat[pickup_train,],Y=cf_bow_type[pickup_train,]), 
                              eval=list(X=ctr_full_title_wordsseq_mat[pickup_eval,],Y=cf_bow_type[pickup_eval,]))
d_seq_purpose_char_class <- list(train=list(X=ctr_full_purposeseq_mat[pickup_train,],Y=cf_bow_class[pickup_train,]), 
                               eval=list(X=ctr_full_purposeseq_mat[pickup_eval,],Y=cf_bow_class[pickup_eval,]))
d_seq_purpose_char_type <- list(train=list(X=ctr_full_purposeseq_mat[pickup_train,],Y=cf_bow_type[pickup_train,]), 
                              eval=list(X=ctr_full_purposeseq_mat[pickup_eval,],Y=cf_bow_type[pickup_eval,]))
d_seq_purpose_word_class <- list(train=list(X=ctr_full_purpose_wordsseq_mat[pickup_train,],Y=cf_bow_class[pickup_train,]), 
                               eval=list(X=ctr_full_purpose_wordsseq_mat[pickup_eval,],Y=cf_bow_class[pickup_eval,]))
d_seq_purpose_word_type <- list(train=list(X=ctr_full_purpose_wordsseq_mat[pickup_train,],Y=cf_bow_type[pickup_train,]), 
                              eval=list(X=ctr_full_purpose_wordsseq_mat[pickup_eval,],Y=cf_bow_type[pickup_eval,]))

save(d_bow_title_char_class, file=paste(DATADIR,"/","d_bow_title_char_class.RData",sep=""))
save(d_bow_title_char_type,  file=paste(DATADIR,"/","d_bow_title_char_type.RData",sep=""))
save(d_bow_title_word_class, file=paste(DATADIR,"/","d_bow_title_word_class.RData",sep=""))
save(d_bow_title_word_type,  file=paste(DATADIR,"/","d_bow_title_word_type.RData",sep=""))
save(d_bow_purpose_char_class, file=paste(DATADIR,"/","d_bow_purpose_char_class.RData",sep=""))
save(d_bow_purpose_char_type,  file=paste(DATADIR,"/","d_bow_purpose_char_type.RData",sep=""))
save(d_bow_purpose_word_class, file=paste(DATADIR,"/","d_bow_purpose_word_class.RData",sep=""))
save(d_bow_purpose_word_type,  file=paste(DATADIR,"/","d_bow_purpose_word_type.RData",sep=""))
if(WITH_ONEHOT) {
  save(d_ohm_title_char_class, file=paste(DATADIR,"/","d_ohm_title_char_class.RData",sep=""))
  save(d_ohm_title_char_type,  file=paste(DATADIR,"/","d_ohm_title_char_type.RData",sep=""))
  save(d_ohm_title_word_class, file=paste(DATADIR,"/","d_ohm_title_word_class.RData",sep=""))
  save(d_ohm_title_word_type,  file=paste(DATADIR,"/","d_ohm_title_word_type.RData",sep=""))
  save(d_ohm_purpose_char_class, file=paste(DATADIR,"/","d_ohm_purpose_char_class.RData",sep=""))
  save(d_ohm_purpose_char_type,  file=paste(DATADIR,"/","d_ohm_purpose_char_type.RData",sep=""))
  save(d_ohm_purpose_word_class, file=paste(DATADIR,"/","d_ohm_purpose_word_class.RData",sep=""))
  save(d_ohm_purpose_word_type,  file=paste(DATADIR,"/","d_ohm_purpose_word_type.RData",sep=""))
}
save(d_seq_title_char_class, file=paste(DATADIR,"/","d_seq_title_char_class.RData",sep=""))
save(d_seq_title_char_type,  file=paste(DATADIR,"/","d_seq_title_char_type.RData",sep=""))
save(d_seq_title_word_class, file=paste(DATADIR,"/","d_seq_title_word_class.RData",sep=""))
save(d_seq_title_word_type,  file=paste(DATADIR,"/","d_seq_title_word_type.RData",sep=""))
save(d_seq_purpose_char_class, file=paste(DATADIR,"/","d_seq_purpose_char_class.RData",sep=""))
save(d_seq_purpose_char_type,  file=paste(DATADIR,"/","d_seq_purpose_char_type.RData",sep=""))
save(d_seq_purpose_word_class, file=paste(DATADIR,"/","d_seq_purpose_word_class.RData",sep=""))
save(d_seq_purpose_word_type,  file=paste(DATADIR,"/","d_seq_purpose_word_type.RData",sep=""))
     

