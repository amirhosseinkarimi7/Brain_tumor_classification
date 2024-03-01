library(tensorflow)
library(keras)
library(tfdatasets)
library(png)
library(ggplot2)
library(patchwork)

# Set the path to your image directory
data_dir <- "~/Documents/kaggle_datasets/Cancer Image Classification/database"

# Get image count
image_count <- length(list.files(data_dir, pattern = "\\.jpg$", recursive = TRUE))
print(paste("Total Images:", image_count))

# Batch size
batch_size <- 64
# Image height
img_height <- 180
# Image width
img_width <- 180

image_gen_train <- keras::image_data_generator(rescale = 1./255,
                                               rotation_range = 10,
                                               width_shift_range = 0.15,
                                               height_shift_range = 0.15,
                                               horizontal_flip = TRUE,
                                               zoom_range = 0.05,
                                               validation_split = 0.2)

# The validation and test images are not augmented
image_gen_validation <- keras::image_data_generator(rescale = 1./255,
                                                    validation_split = 0.2)  


train_data_gen <- keras::flow_images_from_directory(
  data_dir,
  generator = image_gen_train,
  batch_size = batch_size,
  target_size = c(img_height, img_width),
  class_mode = "categorical",
  subset="training",
  seed = 123,
  
)

validation_data_gen <- keras::flow_images_from_directory(
  data_dir,
  generator = image_gen_validation,
  batch_size = batch_size,
  target_size = c(img_height, img_width),
  class_mode = "categorical",
  seed = 123,
  subset="validation"
)




model <- keras::keras_model_sequential() %>% 
  layer_conv_2d(filters = 16,
                kernel_size = 3,
                padding = "same",
                activation = "relu",
                input_shape = c(img_height,
                                img_width,
                                3)) %>% 
  layer_max_pooling_2d() %>% 
  
  layer_dropout(0.2) %>% 
  layer_conv_2d(filters = 32,
                kernel_size = 3,
                padding = "same",
                activation = "relu") %>% 
  layer_max_pooling_2d() %>% 
  layer_dropout(0.2) %>% 
  layer_flatten() %>% 
  layer_dense(512,
              activation = "relu") %>% 
  layer_dense(1,
              activation = "sigmoid")

model %>% summary()

model %>% compile(loss = "binary_crossentropy",
                  optimizer = optimizer_adam(),
                  metrics = c("accuracy"))

history <- keras::fit(model,
                                train_data_gen,
                                steps_per_epoch = floor(total_train / batch_size),
                                epochs = 10,
                                validation_data = validation_data_gen,
                                validation_steps = floor(total_validation / batch_size),
                                callbacks = callback_early_stopping(monitor = "val_loss",
                                                                    min_delta = 0.01,
                                                                    patience = 4))






