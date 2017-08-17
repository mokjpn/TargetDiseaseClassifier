
# This is the server logic for a Shiny web application.
# You can find out more about building applications with Shiny here:
#
# http://shiny.rstudio.com
#

library(keras)
library(shiny)
options(shiny.sanitize.errors = TRUE)

load("model_nn.RData")
load("Eval10000.RData")
source("DataUtils.R")
model_nn <- unserialize_model(model_nn_serialized)


shinyServer(function(input, output) {
  output$class <- renderText({
    eval_title(model_nn, input$text, ohmasters$categoryMaster, ohmasters$titlesCharactersMaster, ohmasters$max_title_length)
  })

})
