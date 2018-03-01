Classify target disease of clinical trial project registerred in UMIN-CTR Clinical Trial Registry
================

What is this?
-------------

This is an experimental machine learning project, to train some neural networks to classify texts in the clinical trial registries automatically.

Usage
-----

1.  Download UMIN-CTR comma-separated text archive.

Currently UMIN-ID is required to download the archive. Download files including Japanese, from <http://www.umin.ac.jp/ctr/csvdata.html>

1.  Prepare running environment.

You will need R, {readr} package, {RMeCab} package and {keras} package. RMeCab is widely used for Japanese text mining. To install, see <http://rmecab.jp/wiki/index.php?RMeCab>

1.  Generate dataset for training and evaluation.

Source Dataset.R in this repository in your R. This generates datasets and serialization of text tokenizers into "data/" or "data\_large" directory. When you set "WITH\_ONEHOT" variable to TRUE, it generates data under "data/" directory. This contains one-hot vector representation of texts (very large file), but used texts will be small (only 300 entries for each of training and evaluation dataset) If you set it to FALSE, the data will be generated under "data\_large/" directory. 10000 entries will be used for each of training and evaluation dataset.

``` r
source("Dataset.R")
```

This code will generate some datasets, which is round-robin combination of "data source", "word separation method", "word embedding method", and "prediction target". There are two data source, research title ("title") and research objective ("purpose"). And two word separation method, character-level separation("char") or word separation powerred by RMeCab ("word"). Number of word embedding methods are three: Bag of Words ("bow"), One-Hot matrix("ohm"), and sequence of word/character code("seq"). Prediction target will be selected from two candidates: target disease classification ("class") or general study type ("type", one of intervation, observation or others).

So there will be 2x2x3x2=24 data files under "data" directory when you set "WITH\_ONEHOT" to TRUE. If that is set to FALSE, "ohm" will be omitted, so there will be 2x2x2x2=16 data files under "data\_large" directory.

1.  Train the model.

Source train.R in this repository in your R. To run code without any modification, you need both of "data" and "data\_large" datasets.

``` r
source("train.R")
```

This will train each dataset by 3 types of deep neural network: simple deep neural network with three hidden layer ("dense"), recurrent neural network ("rnn"), and the network with 1-dimension convolutional layer("1dc"). Trained models will be saved under "model" directory and "model\_large" directory.

1.  Try the trained model.

As the dataset was created, you also have text tokenizers file under "data" and "data\_large" directory. With this, you can test classification by giving some new text.

``` r
library(keras)
library(RMeCab)
library(reticulate)
source("DataUtils.R")
load("data/PredictionMasters.RData") # also created in Dataset.R
model <- load_model_hdf5("model/m_bow_purpose_word_class_dense.hdf5")
tokenizer <- load_text_tokenizer("data/purpose_word_tokenizer.keras")
eval_text("SGLT-2阻害薬の長期的な投与の合併症罹患への影響について検討を行う。", model, tokenizer, master=DiseaseClassesMaster, word=TRUE, bow=TRUE)
```

    ## [1] "内分泌・代謝病内科学/Endocrinology and Metabolism"

``` r
model <- load_model_hdf5("model/m_seq_title_char_type_1dc.hdf5")
tokenizer <- load_text_tokenizer("data/title_char_tokenizer.keras")
eval_text("在宅医療における転倒リスク要因の探索", model, tokenizer, master=TypesMaster, word=FALSE, bow=FALSE)
```

    ## [1] "観察/Observational"
