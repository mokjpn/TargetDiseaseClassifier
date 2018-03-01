
# This is the user-interface definition of a Shiny web application.
# You can find out more about building applications with Shiny here:
#
# http://shiny.rstudio.com
#

library(shiny)

shinyUI(fluidPage(

  # Application title
  titlePanel("Demonstration of Text Classification"),
  p("UMIN-CTRのデータを学習させたニューラルネットワークを用いて、文章から関連する医学分野および研究の種類を推定します。"),
  a(href="https://github.com/mokjpn/TargetDiseaseClassifier/", "詳細はこちら"),
  textAreaInput("text",  "研究目的を入力してください"),
  selectInput("kakasi", "利用するモデルの単語分割方式を選択してください", choices=c("文字単位"="char", "単語単位"="word"), selected="word"),
  selectInput("embed", "利用するモデルの文字埋め込み方式を選択してください", choices=c("Bag of Words"="bow", "Embedding Layer"="seq"), selected="bow"),
  selectInput("network", "利用するモデルの構造を選択してください", choices=c("全結合"="dense", "RNN"="rnn", "1次元畳み込み"="1dc"), selected="dense"),
  actionButton("eval", "推定"),
  p("推定される分野："), textOutput("class"),
  p("推定される種別："), textOutput("type")
)
)
