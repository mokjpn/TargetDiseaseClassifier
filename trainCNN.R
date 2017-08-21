library(keras)

load("Train15000E.RData") # contains ohmasters, train_data
X <- train_data$X
Y <- train_data$Y

input <- layer_input(shape=dim(X)[2])
embedding_size <- 1500
filter_size <- 32
filters <- c(2,3,4,5)

embedded <- layer_embedding(input, length(ohmasters$titlesCharactersMaster), embedding_size, input_length=ohmasters$max_title_length) 
  
filter_layers <- list()
for(flnum in filters) {
  oneD <- embedded
  oneD <- layer_conv_1d( oneD, filter_size,flnum,activation = 'relu')
  oneD <- layer_max_pooling_1d(oneD)
  filter_layers <- append(filter_layers, oneD)
}
convs <- layer_concatenate(filter_layers, axis=1)
convs <- layer_flatten(convs)

out <- layer_dense(convs, units=1500, activation = 'relu')
out <- layer_dropout(out, rate=0.1)
out <- layer_dense(convs, units=500, activation = 'relu')
out <- layer_dropout(out, rate=0.1)
out <- layer_dense(out,units=dim(Y)[2], activation='sigmoid')

model_cnn <- keras_model(input, out)
model_cnn %>% compile(loss='binary_crossentropy',optimizer='adam',metrics='accuracy')

history_cnn <- model_cnn %>% fit(X, Y, epochs=25, batch_size=200)

load("Eval10000E.RData") # contains ohmasters, eval_data
loss_and_metrics_cnn <- model_cnn %>% evaluate(eval_data$X, eval_data$Y)
eval_multilabel_cnn <- eval_multilabel(model_cnn, eval_data)

model_cnn_serialized <- serialize_model(model_cnn)
save(model_cnn_serialized, file="model_cnn.RData")
save_model_hdf5(model_cnn,"model_cnn.hdf5") 


