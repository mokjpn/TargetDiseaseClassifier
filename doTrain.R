
doTrain <- function(source=c("title", "purpose"), tokenize=c("char", "word"), embed=c("bow", "ohm", "seq"),
                             target=c("class", "type"), network=c("dense", "rnn", "1dc"), large=FALSE) {
  SOURCE=match.arg(source)
  TOKENIZE=match.arg(tokenize)
  EMBED=match.arg(embed)
  TARGET=match.arg(target)
  
  NETWORK=match.arg(network)

  if(EMBED == "bow" && NETWORK != "dense")
    stop("Bag of Words can only be fit for simple NN.")
  
  varname=paste("d",EMBED,SOURCE,TOKENIZE,TARGET,sep="_")
  if(large) {
    datadir="data_large"
    modeldir="model_large"  
  }
  else {
    datadir="data"
    modeldir="model"  
  }
  datafile=paste(datadir,"/", varname,".RData",sep="")
  load(datafile)
  data <- eval(parse(text=varname))
  name <- paste("m", EMBED, SOURCE, TOKENIZE, TARGET, NETWORK, sep="_")

  X <- data$train$X
  Y <- data$train$Y

  modelfile <- paste(modeldir, "/", name,".hdf5",sep="")
  if(file.exists(modelfile)) {
    model <- load_model_hdf5(modelfile)
  } else {
  
  switch (EMBED,
      "bow" = {
        inputs <- layer_input(shape=dim(X)[2])
        embed <- inputs
      },
      "ohm" = {
        inputs <- layer_input(shape=c(dim(X)[2],dim(X)[3]))
        embed <- inputs
        if(NETWORK == "dense")
          embed <- embed %>% layer_flatten()
      },
      "seq" = {
        nfeat <- max(c(as.vector(data$train$X), as.vector(data$eval$X)))
        inputs <- layer_input(shape=dim(X)[2])
        embed <- inputs %>% layer_embedding(input_dim=nfeat,output_dim=64)
        if(NETWORK == "dense")
          embed <- embed %>% layer_flatten()
      })
  
  switch(NETWORK, 
         "dense" = {
           predictions <- embed %>%
             layer_dense(units=100, activation='relu')  %>%
             layer_dense(units=50, activation = 'relu') %>%
             layer_dense(units=20, activation = 'relu') %>%
             layer_dense(units=dim(Y)[2], activation='sigmoid')
         }, 
         "rnn" = {
           predictions <- embed %>%
             layer_simple_rnn(units = 100, return_sequences = TRUE) %>% 
             layer_simple_rnn(units = 100, return_sequences = TRUE) %>% 
             layer_simple_rnn(units = 100, return_sequences = TRUE) %>%
             layer_simple_rnn(units = 100) %>% 
             layer_dense(units=dim(Y)[2], activation='sigmoid')
         }, 
         "1dc" = {
           predictions <- embed %>%
             layer_conv_1d(filters=32, kernel_size=7, activation="relu") %>%
             layer_max_pooling_1d(pool_size=5) %>%
             layer_conv_1d(filters=32, kernel_size=7, activation="relu") %>%
             layer_global_max_pooling_1d() %>%
             layer_dense(units=dim(Y)[2], activation='sigmoid')
         }
  )
  
  model <- keras_model(inputs=inputs, outputs = predictions)
  
  model %>% compile(loss='binary_crossentropy',optimizer='rmsprop',metrics='categorical_accuracy')
  
  history <- model %>% fit(X, Y, epochs=100)
  save_model_hdf5(model,paste(modeldir, "/", name,".hdf5",sep=""))
  }
  
  loss_and_metrics <- model %>% evaluate(data$eval$X, data$eval$Y)
  eval_multilabel_identical <- eval_multilabel(model, data$eval, method="maxidentical")
  eval_multilabel_rank <- eval_multilabel(model, data$eval, rankmax = 5, method="rank")
  return(list(name=name, loss_and_metrics=loss_and_metrics, 
              eval_multilabel_identical=eval_multilabel_identical,
              eval_multilabel_rank=eval_multilabel_rank))
}
