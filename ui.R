
# This is the user-interface definition of a Shiny web application.
# You can find out more about building applications with Shiny here:
#
# http://shiny.rstudio.com
#

library(shiny)

shinyUI(fluidPage(

  # Application title
  titlePanel("Demonstration of Text Classification"),
  p("UMIN-CTRのデータを学習させたニューラルネットワークを用いて、文章から関連する医学分野を推定します。"),
  a(href="https://github.com/mokjpn/TargetDiseaseClassifier/", "詳細はこちら"),
  textAreaInput("text",  "研究目的を入力してください"),
  submitButton("推定"),
  p("推定される分野："), textOutput("class")
    )
)
