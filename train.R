library(keras)

load("Train15000.RData") # contains ohmasters, train_data
X <- train_data$X
Y <- train_data$Y

model_cnn <- keras_model_sequential()

model_cnn %>% 
  layer_dense(units=1500, activation='relu', input_shape=dim(X)[2]) %>%
  layer_dropout(rate=0.1) %>% 
  layer_dense(units=500, activation = 'relu') %>%
  layer_dropout(rate=0.1) %>%
  layer_dense(units=dim(Y)[2], activation='sigmoid')

#sgd <- optimizer_sgd(lr=0.01, decay=1e-6, momentum=0.9, nesterov=TRUE)
model_cnn %>% compile(loss='binary_crossentropy',optimizer='adam',metrics='accuracy')

history_cnn <- model_cnn %>% fit(X, Y, epochs=100, batch_size=2000)

load("Eval10000.RData") # contains ohmasters, eval_data
loss_and_metrics_cnn <- model_cnn %>% evaluate(eval_data$X, eval_data$Y)

