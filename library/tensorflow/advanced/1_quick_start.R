# ***************************************************************************************
# Title     : Quick Start
# Objective : TODO
# Created by: Owner
# Created on: 2020/10/18
# URL       : https://tensorflow.rstudio.com/tutorials/advanced/
# ***************************************************************************************


# ＜ポイント＞
# - インプットデータの前処理（tfdatasets）
# - カスタムモデルの構築
# - カスタムトレーニングのループ（tfautograph）


# ＜参考＞
# tfautograph
# https://t-kalinowski.github.io/tfautograph/articles/tfautograph.html



# ＜目次＞
# 0 環境準備
# 1 データ加工
# 2 モデル構築
# 3 モデル評価
# 4 ループ処理の定義
# 5 ループトレーニング



# 0 環境準備 -------------------------------------------------------------------------------

# ライブラリ
library(reticulate)
library(tidyverse)
library(keras)
library(tfdatasets)
library(tfautograph)
library(purrr)


# 仮想環境の選択
use_condaenv("C:/Users/Owner/Anaconda3/envs/r-reticulate", required = TRUE)
py_config()


# データ準備
# --- MNISTデータ
mnist <- dataset_mnist()


# データ変換
# --- 3つのピクセルの値は0〜255の整数であり、0〜1の浮動小数点数に変換
mnist$train$x <- mnist$train$x / 255
mnist$test$x <- mnist$test$x / 255


# 次元を追加
dim(mnist$train$x) <- c(dim(mnist$train$x), 1)
dim(mnist$test$x)  <- c(dim(mnist$test$x), 1)


# データ構造
mnist %>% glimpse()



# 1 データ加工 -------------------------------------------------------------------------------

# 訓練データ
# --- データの一部を抽出してシャッフル
train_ds <-
  mnist$train %>%
    tensor_slices_dataset() %>%
    dataset_take(20000) %>%
    dataset_map(~modify_at(.x, "x", tf$cast, dtype = tf$float32)) %>%
    dataset_map(~modify_at(.x, "y", tf$cast, dtype = tf$int64)) %>%
    dataset_shuffle(10000) %>%
    dataset_batch(32)


# テストデータ
# --- データの一部を抽出してシャッフル
test_ds <-
  mnist$test %>%
    tensor_slices_dataset() %>%
    dataset_take(2000) %>%
    dataset_map(~modify_at(.x, "x", tf$cast, dtype = tf$float32)) %>%
    dataset_map(~modify_at(.x, "y", tf$cast, dtype = tf$int64)) %>%
    dataset_batch(32)


# 確認
train_ds %>% print()
test_ds %>% print()



# 2. モデル構築 -------------------------------------------------------------------------------

# 関数定義
# --- モデル構築
simple_conv_nn <- function(filters, kernel_size) {
  keras_model_custom(name = "MyModel", function(self) {

    # モデル部品
    self$conv1 <- layer_conv_2d(
      filters = filters,
      kernel_size = rep(kernel_size, 2),
      activation = "relu"
    )

    self$flatten <- layer_flatten()

    self$d1 <- layer_dense(units = 128, activation = "relu")
    self$d2 <- layer_dense(units = 10, activation = "softmax")


    # モデル構築
    function(inputs, mask = NULL) {
      inputs %>%
        self$conv1() %>%
        self$flatten() %>%
        self$d1() %>%
        self$d2()
    }
  })
}


# モデル構築
model <- simple_conv_nn(filters = 32, kernel_size = 3)



# 3. トレーニング設定 -------------------------------------------------------------------------------

# その他の設定
# --- 損失関数
# --- オプティマイザ
loss <- loss_sparse_categorical_crossentropy
optimizer <- optimizer_adam()


# 訓練データ
train_loss     <- tf$keras$metrics$Mean(name='train_loss')
train_accuracy <- tf$keras$metrics$SparseCategoricalAccuracy(name='train_accuracy')


# テストデータ
test_loss     <- tf$keras$metrics$Mean(name='test_loss')
test_accuracy <- tf$keras$metrics$SparseCategoricalAccuracy(name='test_accuracy')



# 関数定義
# --- トレーニング・ステップ
train_step <- function(images, labels) {

  with (tf$GradientTape() %as% tape, {
    predictions <- model(images)
    l <- loss(labels, predictions)
  })

  gradients <- tape$gradient(l, model$trainable_variables)
  optimizer$apply_gradients(purrr::transpose(list(
    gradients, model$trainable_variables
  )))

  train_loss(l)
  train_accuracy(labels, predictions)

}


# 関数定義
# --- テスト・ステップ
test_step <- function(images, labels) {
  predictions <- model(images)
  l <- loss(labels, predictions)

  test_loss(l)
  test_accuracy(labels, predictions)
}



# 4. ループ処理の定義 -------------------------------------------------------------------------------

# 関数定義
# ---
training_loop <- tf_function(autograph(function(train_ds, test_ds) {

  for (b1 in train_ds) {
    train_step(b1$x, b1$y)
  }

  for (b2 in test_ds) {
    test_step(b2$x, b2$y)
  }

  tf$print("Acc", train_accuracy$result(), "Test Acc", test_accuracy$result())

  train_loss$reset_states()
  train_accuracy$reset_states()
  test_loss$reset_states()
  test_accuracy$reset_states()

}))



# 5. ループトレーニング -------------------------------------------------------------------------------

# ループトレーニング
# --- 5つのエポックのトレーニングループを実行
for (epoch in 1:5) {
  cat("Epoch: ", epoch, " -----------\n")
  training_loop(train_ds, test_ds)
}
