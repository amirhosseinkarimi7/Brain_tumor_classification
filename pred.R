library(tensorflow)
library(keras)
library(tidyverse)
library(reticulate)

# Load the pre-trained model
model <- keras::load_model_hdf5("brain_tumor_classification.keras")

# Example usage
image_path="~/Documents/kaggle_datasets/Cancer Image Classification/testnotumor.png"


prediction <- image_load(image_path, target_size = c(180, 180)) %>%
  image_to_array() %>%
  array_reshape(dim = c(1, 180, 180, 3)) %>%  # Add batch dimension (assuming 3 channels)
  model$predict()%>% 
  { exp(.) / sum(exp(.)) }


# Get the class name corresponding to the highest prediction
class_index <- which.max(prediction)
class_name <- switch(class_index,
                    "1" = "glioma",
                    "2" = "meningioma",
                    "3" = "notumor",
                    "4" = "pituitary")

score <- prediction[which.max(prediction)] 
