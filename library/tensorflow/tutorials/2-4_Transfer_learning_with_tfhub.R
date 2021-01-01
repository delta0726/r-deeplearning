# ***************************************************************************************
# Title     : Transfer learning with tfhub
# Objective : TODO
# Created by: Owner
# Created on: 2020/10/21
# URL       : https://tensorflow.rstudio.com/tutorials/beginners/basic-ml/tutorial_basic_text_classification_with_tfhub/
# ***************************************************************************************


# ＜目的＞
# - tfhubの学習済モデルを用いた移転学習
#   --- エラーが発生するため見合わせ



# 0 準備 -------------------------------------------------------------------------------

# ライブラリ
library(reticulate)
library(keras)
library(tfhub)
library(tfds)
library(tfdatasets)


# 仮想環境の選択
use_condaenv("C:/Users/Owner/Anaconda3/envs/r-reticulate", required = TRUE)
py_config()


# 1 データ取得 -------------------------------------------------------------------------

# データ準備
# --- IMDBデータセットを使用
# --- インターネット映画データベースからの50,000本の映画レビューのテキストデータ
imdb <- tfds_load(
  "imdb_reviews:1.0.0",
  split = list("train[:60%]", "train[-40%:]", "test"),
  as_supervised = TRUE
)
#imdb <- dataset_imdb(num_words = 10000)
#c(c(train_data, train_label), c(test_data, test_label)) %<-% imdb

# エラー発生
#  Please use the plain text version with `tensorflow_text`.



# ---------------------------------------------------------------------------------


#
## データ確認
#imdb %>% summary()
#
#
#
#
## 0. 準備 -------------------------------------------------------------------------------
#
#first <- imdb[[1]] %>%
#  dataset_batch(1) %>% # Used to get only the first example
#  reticulate::as_iterator() %>%
#  reticulate::iter_next()
#str(first)
#
#
#
#embedding_layer <- layer_hub(handle = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1")
#embedding_layer(first[[1]])
#
#model <- keras_model_sequential() %>%
#  layer_hub(
#    handle = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1",
#    input_shape = list(),
#    dtype = tf$string,
#    trainable = TRUE
#  ) %>%
#  layer_dense(units = 16, activation = "relu") %>%
#  layer_dense(units = 1, activation = "sigmoid")
#
#
#
#
#model %>%
#  compile(
#    optimizer = "adam",
#    loss = "binary_crossentropy",
#    metrics = "accuracy"
#  )
#
#model %>%
#  fit(
#    imdb[[1]] %>% dataset_shuffle(10000) %>% dataset_batch(512),
#    epochs = 20,
#    validation_data = imdb[[2]] %>% dataset_batch(512),
#    verbose = 2
#  )
#
#model %>%
#  evaluate(imdb[[3]] %>% dataset_batch(512), verbose = 0)
