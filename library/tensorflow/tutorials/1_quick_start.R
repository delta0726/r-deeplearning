# ***************************************************************************************
# Title     : Quick Start
# Objective : TODO
# Created by: Owner
# Created on: 2020/11/18
# URL       : https://tensorflow.rstudio.com/tutorials/beginners/
# ***************************************************************************************


# ＜ポイント＞
# - Anacondaの仮想環境を用いてPythonライブラリを操作している
#   --- 仮想環境の管理はcondaコマンドで行うのが無難
# - モデリングの箇所は完全にPythonオブジェクトとなっている
# - モデル構築全般がRらしくない書き方(副作用あり)になっている


# ＜目次＞
# 0 環境準備
# 1 データ準備
# 2 モデル構築
# 3 モデル訓練
# 4 モデル評価
# 5 モデル保存



# 0 環境準備 -------------------------------------------------------------------------------

# ライブラリ
library(reticulate)
library(tidyverse)
library(keras)


# 仮想環境の選択
use_condaenv("C:/Users/Owner/Anaconda3/envs/r-reticulate", required = TRUE)
py_config()



# 1 データ準備 -------------------------------------------------------------------------------

# データ準備
# --- MNISTデータ
mnist <- dataset_mnist()


# データ変換
# --- 3つのピクセルの値は0〜255の整数であり、0〜1の浮動小数点数に変換
mnist$train$x <- mnist$train$x / 255
mnist$test$x <- mnist$test$x / 255


# データ構造
mnist %>% glimpse()



# 2 モデル構築 -------------------------------------------------------------------------------

# ネットワーク構築
# --- 最初のレイヤーinput_shapeは入力の次元を表す引数を指定する必要がある
# --- 今回の場合、画像は28x28です。
model <-
  keras_model_sequential() %>%
    layer_flatten(input_shape = c(28, 28)) %>%
    layer_dense(units = 128, activation = "relu") %>%
    layer_dropout(0.2) %>%
    layer_dense(10, activation = "softmax")


# サマリー
model %>% summary()


# クラス確認
model %>% class()


# コンパイル
# --- ｢損失関数｣｢オプティマイザー｣｢評価指標｣を指定する
# --- Rには珍しい副作用なしの操作だが、modeオブジェクトには反映している
model %>%
  compile(
    loss = "sparse_categorical_crossentropy",
    optimizer = "adam",
    metrics = "accuracy"
  )



# 3 モデル訓練 -------------------------------------------------------------------------------

# トレーニング実行
# --- ｢エポック数｣｢評価比率｣などを設定
model %>%
  fit(
    x = mnist$train$x, y = mnist$train$y,
    epochs = 5,
    validation_split = 0.3,
    verbose = 2
  )




# 3 モデル評価 -------------------------------------------------------------------------------

# メトリック出力
model %>%
  evaluate(mnist$test$x, mnist$test$y, verbose = 0)




# 4 モデル保存 -------------------------------------------------------------------------------

# モデル保存
save_model_tf(object = model, filepath = "library/tensorflow/tutorials/model")


# モデルロード
reloaded_model <- load_model_tf("library/tensorflow/tutorials/model")


# 比較
all.equal(predict(model, mnist$test$x), predict(reloaded_model, mnist$test$x))

