library(shiny)
library(tensorflow)
library(keras)
library(reticulate)
library(tidyverse)


server <- function(input, output) {
  
  model <- keras::load_model_hdf5("brain_tumor_classification.keras")
  #setwd("~/Documents/kaggle_datasets/Cancer Image Classification")
  
  
  # Load the pre-trained model
  model <- keras::load_model_hdf5("brain_tumor_classification.keras")
  
  # Function to process image and make prediction
  predict_image <- function(image_path) {
    prediction <- image_load(image_path, target_size = c(180, 180)) %>%
      image_to_array() %>%
      array_reshape(dim = c(1, 180, 180, 3)) %>% # Add batch dimension (assuming 3 channels)
      model$predict() %>%
      { exp(.) / sum(exp(.)) }
    
    # Get the class name corresponding to the highest prediction
    class_index <- which.max(prediction)
    class_name <- switch(class_index,
                         "1" = 'glioma',
                         "2" = 'meningioma',
                         "3" = 'notumor',
                         "4" = 'pituitary')
    
    score <- prediction[which.max(prediction)]
    
    return(list(class_name = class_name, score = score))
  }
  
  # When the predict button is clicked
  observeEvent(input$predict, {
    req(input$file)
    
    # Get the path of the uploaded file
    img <- input$file$datapath
    
    # Make prediction
    result <- predict_image(img)
    
    # Update outputs
    output$class_name <- renderText({
      paste("Predicted Class: ", result$class_name)
    })
    output$score <- renderText({
      paste("Confidence Score: ", result$score)
    })
    
    # Display the uploaded image
    output$uploaded_image <- renderImage({
      list(src = img, contentType = 'image/png', width = 300)
    })
  })
}



