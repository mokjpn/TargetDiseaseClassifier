
# This is the server logic for a Shiny web application.
# You can find out more about building applications with Shiny here:
#
# http://shiny.rstudio.com
#

library(shiny)
library(keras)
library(RMeCab)
library(reticulate)

options(shiny.sanitize.errors = TRUE)

source("DataUtils.R")
load("data_large/PredictionMasters.RData")

shinyServer(function(input, output) {
  modelspec <- reactive({
    validate(
      need( input$embed != "bow" || (input$embed == "bow" && input$network == "dense"), "Bag of Wordsの時はRNNと畳み込みは利用できません")
    )
    m1 <- paste("model_large/m_", switch(input$embed,
            "bow"="bow",
            "seq"="seq"), "_purpose_",
          switch(input$kakasi,
            "char"="char",
            "word"="word"), "_", sep="")
    m2 <- paste("_", switch(input$network,
            "dense"="dense",
            "rnn"="rnn",
            "1dc"="1dc"),".hdf5",sep="")
    list(class=paste(m1,"class", m2,sep=""),
         type=paste(m1,"type",m2,sep=""))
  })
  tokenizerspec <- reactive({
    paste("data_large/purpose_",
          switch(input$kakasi, 
                 "char"="char",
                 "word"="word"), "_tokenizer.keras", sep="")
  })

  isword <- reactive({
    switch(input$kakasi,
           "char"=FALSE,
           "word"=TRUE)    
  })
  isbow <- reactive({
    switch(input$embed,
           "bow"=TRUE,
           FALSE)    
  })
  
  output$class <- renderText({
    input$eval
    model <- isolate(load_model_hdf5(modelspec()$class))
    tokenizer <- isolate(load_text_tokenizer(tokenizerspec()))
    if(isolate(input$text)=="")
      "判定対象なし"
    else
      isolate(eval_text(input$text, model, tokenizer, master=DiseaseClassesMaster, word=isword(), bow=isbow()))
  })    
  output$type <- renderText({
    input$eval
    model <- isolate(load_model_hdf5(modelspec()$type))
    tokenizer <- isolate(load_text_tokenizer(tokenizerspec()))
    if(isolate(input$text)=="")
      "判定対象なし"
    else
      isolate(eval_text(input$text, model, tokenizer, master=TypesMaster, word=isword(), bow=isbow()))
  })

})

