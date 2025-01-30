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
network0 %>% compile(optimizer = "rmsprop", 
                     loss = "categorical_crossentropy", 
                     metrics = c("accuracy", "mse"))

# Train the base neural network and save into history object.
hist0 <- network0 %>% fit(train_images, train_labels, epochs = 5,
                          batch_size = 128)
plot(hist0)

# Assess the base network using the test data.
results0 <- network0 %>% evaluate(test_images, test_labels)
results0

#### Create network with different epoch number. ####
# Define, compile and assess the network.
network1 <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = "relu", 
              input_shape = (28 * 28)) %>%
  layer_dense(units = 10, activation = "softmax")
network1
network1 %>% compile(optimizer = "rmsprop", 
                     loss = "categorical_crossentropy", 
                     metrics = c("accuracy", "mse"))
hist1 <- network1 %>% fit(train_images, train_labels, epochs = 15,
                          batch_size = 128)
results1 <- network1 %>% evaluate(test_images, test_labels)
results1
