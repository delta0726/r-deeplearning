# ***************************************************************************************
# Library   : kerastuneR
# Title     : Restrict Search Space
# Created by: Owner
# Created on: 2021/05/07
# URL       : https://github.com/EagerAI/kerastuneR
# ***************************************************************************************


# ＜概要＞
# - ループを使ってモデルのレイヤーを柔軟に変更できるようにする


# ＜参考＞
# Python Keras Tuner Documentation
# - https://keras-team.github.io/keras-tuner/


# ＜目次＞
# 0 準備
# 1 データ作成
# 2 モデル構築
# 3 チューニング
# 4 結果検証


# 0 準備 -------------------------------------------------------------------------------

# ライブラリ
library(reticulate)
library(tidyverse)
library(tfdatasets)
library(keras)
library(tensorflow)
library(kerastuneR)

# 仮想環境の選択
use_condaenv("C:/Users/Owner/Anaconda3/envs/r-reticulate", required = TRUE)
py_config()


# 1 データ作成  --------------------------------------------------------------------------

# データロード
mnist_data = dataset_fashion_mnist()
c(mnist_train, mnist_test) %<-%  mnist_data
rm(mnist_data)

# データ基準化
mnist_train$x = tf$dtypes$cast(mnist_train$x, 'float32') / 255.
mnist_test$x = tf$dtypes$cast(mnist_test$x, 'float32') / 255.

# テンソルに変換
mnist_train$x = keras::k_reshape(mnist_train$x,shape = c(6e4,28,28))
mnist_test$x = keras::k_reshape(mnist_test$x,shape = c(1e4,28,28))


# 2 モデル構築 ----------------------------------------------------------------------------

# ハイパーパラメータ定義
hp = HyperParameters()
hp$Choice('learning_rate', c(1e-1, 1e-3))
hp$Int('num_layers', 2L, 20L)


mnist_model = function(hp) {
  # モデル定義
  model = keras_model_sequential() %>%
    layer_flatten(input_shape = c(28,28))
  # レイヤー作成
  # --- ループで階層作成
  # --- コンパイル
  for (i in 1:(hp$get('num_layers')) ) {
    model %>% layer_dense(32, activation='relu') %>%
      layer_dense(units = 10, activation='softmax')
  } %>%
    compile(
      optimizer = tf$keras$optimizers$Adam(hp$get('learning_rate')),
      loss = 'sparse_categorical_crossentropy',
      metrics = 'accuracy')

  # 出力
  return(model)

}


# 3 チューニング ---------------------------------------------------------------------

# チューナー作成
tuner =
  mnist_model %>%
    RandomSearch(max_trials = 5,
                 hyperparameters = hp,
                 tune_new_entries = T,
                 objective = 'val_accuracy',
                 directory = 'dir_1',
                 project_name = 'mnist_space')

# チューニング
tuner %>%
  fit_tuner(x = mnist_train$x,
            y = mnist_train$y,
            epochs = 5,
            validation_data = list(mnist_test$x, mnist_test$y))


# 4 結果検証 -------------------------------------------------------------------------

# チューナーの表示
# ---
tuner %>% plot_tuner(type = "plotly")
tuner %>% plot_tuner(type = "echarts4r")

# サマリー表示
tuner %>% results_summary(num_trials = 5)

# 上位モデルの選択
best_5_models = tuner %>% get_best_models(num_models = 5)
best_5_models

# モデル構造の表示
best_5_models[[1]] %>%
  plot_keras_model(show_shapes = TRUE, dpi = 96)

