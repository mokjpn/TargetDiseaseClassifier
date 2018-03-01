# Bag of Words vector に変換: 1文章1ベクトル。最終的な次元は、c(例数, カテゴリー数)
to_bow <- function(seqVec, seqMax) {
  return(tabulate(seqVec, nbins=seqMax))
}

## テキストを文字一覧のベクタに変換する関数
generate_charmasters <- function(str) {
  max_length <- max(sapply(str, nchar))
  tc <- strsplit(str, '')
  mas <- levels(factor(unlist(tc)))
  return(list(index=mas, MaxLength=max_length)) # index:文字一覧 MaxLength: 全データ中の最大文字長
}

## テキストを単語一覧のベクタに変換する関数
generate_wordmasters <- function(str, minimumFreq=1) { # minimumFreq回以上出現したものだけをカウント
  v <- vector()
  m <- 0
  sapply(str, function(s) {
    w <- unlist(RMeCabC(s))
    v <<- append(v,w)
    if(length(w) > m) m <<- length(w)
  })
  t <- table(v)
  index <- names(t)[t >= minimumFreq]
  return(list(index=index, MaxLength=m))  # index:単語一覧 MaxLength: 全データ中の最大単語数
}

## 予測結果の評価
### identical: threshould を超える確率をとったものの一覧と正解が完全一致, 
### rank: 確率の上位rankmax位の中に正解が含まれていればOk
### xor: threshould を超える確率をとったものの一覧と正解との食い違いの、全カテゴリ数に対する割合が allowance 未満であればOk
### maxidentical: 確率が最大値となったものが正解と一致すればOk
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

## モデルを与えて単一の文章をカテゴライズさせる
### eval_text requires reticulate package
eval_text <- function(text, model, tokenizer, master, num=1,raw=FALSE,word=TRUE,bow=FALSE) {
  if(word)
    txt <- unlist(RMeCabC(text))
  else
    txt <- strsplit(text,'')[[1]]
  if(bow)
    seq <- t(to_bow(unlist(texts_to_sequences(tokenizer, txt)), model$input_shape[[2]]))
  else
    seq <- pad_sequences(list(unlist(texts_to_sequences(tokenizer, txt))), model$input_shape[[2]])
  pred <- predict(model, seq )
  if(!raw)
    master[order(pred, decreasing = TRUE)[1:num]]
  else {
    o <- order(pred, decreasing = TRUE)[1:num]
    r <- pred[o]
    names(r) <- master[o]
    r
  }
}

