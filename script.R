# Coding Exercise 1

#### Setting Up ####
# Load libraries.
library(keras)
library(ggplot2)

# Build simple neural network.
mnist <- dataset_mnist()
train_images <- mnist$train$x
train_labels <- mnist$train$y
test_images <- mnist$test$x
test_labels <- mnist$test$y

# Pre-processing images, flatten image arrays to vectors and scale.
train_images <- array_reshape(train_images, c(60000, 28 * 28))
train_images <- train_images / 255
test_images <- array_reshape(test_images, c(10000, 28 * 28))
test_images <- test_images / 255

# Pre-processing labels, one hot encoding.
train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)

# Shuffle the training data.
set.seed(123)
I <- sample.int(nrow(train_images))
train_images <- train_images[I,]
train_labels <- train_labels[I,]

#### Create simple (base) neural network. ####
# Define the network.
network0 <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = "relu", 
              input_shape = (28 * 28)) %>%
  layer_dense(units = 10, activation = "softmax")
network0

# Compile the base neural network.
network0 %>% compile(optimizer = "sgd", 
                     loss = "categorical_crossentropy", 
                     metrics = c("accuracy", "mse"))

# Train the base neural network and save into history object.
hist0 <- network0 %>% fit(train_images, train_labels, epochs = 10,
                          batch_size = 128, view_metrics=T, 
                          verbose = 1, validation_split=0.2)
plot(hist0)

# Assess the base network using the test data.
results0 <- network0 %>% evaluate(test_images, test_labels)
results0

#### Create networks with increased epoch numbers. ####
# Define, compile and assess the networks.
# 15 epochs.
network_epoch1 <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = "relu", 
              input_shape = (28 * 28)) %>%
  layer_dense(units = 10, activation = "softmax")
network_epoch1 %>% compile(optimizer = "sgd", 
                     loss = "categorical_crossentropy", 
                     metrics = c("accuracy", "mse"))
hist_epoch1 <- network_epoch1 %>% fit(train_images, train_labels, 
                                     epochs = 15, batch_size = 128, view_metrics=T, 
                                     verbose = 1, validation_split=0.2)
results_epoch1 <- network_epoch1 %>% evaluate(test_images, test_labels)

# 20 epochs.
network_epoch2 <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = "relu", 
              input_shape = (28 * 28)) %>%
  layer_dense(units = 10, activation = "softmax")
network_epoch2 %>% compile(optimizer = "sgd", 
                          loss = "categorical_crossentropy", 
                          metrics = c("accuracy", "mse"))
hist_epoch2 <- network_epoch2 %>% fit(train_images, train_labels, 
                                     epochs = 20, batch_size = 128, view_metrics=T, 
                                     verbose = 1, validation_split=0.2)
results_epoch2 <- network_epoch2 %>% evaluate(test_images, test_labels)

# Compare.
results0
results_epoch1
results_epoch2
hist0
hist_epoch1
hist_epoch2
plot(hist0)
plot(hist_epoch1)
plot(hist_epoch2)

#### Create networks with larger batch sizes. ####
# Batch size of 28 * 2. 
network_bs1 <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = "relu", 
              input_shape = (28 * 28)) %>%
  layer_dense(units = 10, activation = "softmax")
network_bs1 %>% compile(optimizer = "sgd", 
                     loss = "categorical_crossentropy", 
                     metrics = c("accuracy", "mse"))
hist_bs1 <- network_bs1 %>% fit(train_images, train_labels, epochs = 5,
                          batch_size = 128 * 2, view_metrics = T, 
                          verbose = 1, validation_split = 0.2)
results_bs1 <- network_bs1 %>% evaluate(test_images, test_labels)

# Batch size of 28 * 3. 
network_bs2 <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = "relu", 
              input_shape = (28 * 28)) %>%
  layer_dense(units = 10, activation = "softmax")
network_bs2 %>% compile(optimizer = "sgd", 
                        loss = "categorical_crossentropy", 
                        metrics = c("accuracy", "mse"))
hist_bs2 <- network_bs2 %>% fit(train_images, train_labels, epochs = 5,
                                batch_size = 128 * 3, view_metrics = T, 
                                verbose = 1, validation_split = 0.2)
results_bs2 <- network_bs2 %>% evaluate(test_images, test_labels)

# Compare.
results0
results_bs1
results_bs2
hist0
hist_bs1
hist_bs2
plot(hist0)
plot(hist_bs1)
plot(hist_bs2)

#### Create networks with drop out. ####
# Drop out rate of 0.3.
network_do1 <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = "relu", 
              input_shape = (28 * 28)) %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = "softmax")
network_do1 %>% compile(optimizer = "sgd", 
                     loss = "categorical_crossentropy", 
                     metrics = c("accuracy", "mse"))
hist_do1 <- network_do1 %>% fit(train_images, train_labels, epochs = 5,
                                batch_size = 128, view_metrics=T, 
                                verbose = 1, validation_split=0.2)
results_do1 <- network_do1 %>% evaluate(test_images, test_labels)

# Drop out rate of 0.8.
network_do2 <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = "relu", 
              input_shape = (28 * 28)) %>%
  layer_dropout(rate = 0.8) %>%
  layer_dense(units = 10, activation = "softmax")
network_do2 %>% compile(optimizer = "sgd", 
                       loss = "categorical_crossentropy", 
                       metrics = c("accuracy", "mse"))
hist_do2 <- network_do2 %>% fit(train_images, train_labels, epochs = 5,
                             batch_size = 128, view_metrics=T, 
                             verbose = 1, validation_split=0.2)
results_do2 <- network_do2 %>% evaluate(test_images, test_labels)

# Compare.
results0
results_do1
results_do2
hist0
hist_do1
hist_do2
plot(hist0)
plot(hist_do1)
plot(hist_do2)