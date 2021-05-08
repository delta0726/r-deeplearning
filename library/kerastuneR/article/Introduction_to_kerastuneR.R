# ***************************************************************************************
# Library   : kerastuneR
# Title     : Introduction to kerastuneR
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


# 0 準備 --------------------------------------------------------------------------------

# ライブラリ
library(reticulate)
library(tidyverse)
library(tfdatasets)
library(keras)
library(tensorflow)
library(kerastuneR)

# 仮想環境の選択
use_condaenv("C:/Users/Owner/Anaconda3/envs/r-reticulate")
py_config()


# 1 データ作成 ----------------------------------------------------------------------------

# データ作成
# --- 訓練データ
x_data <- matrix(data = runif(500,0,1),nrow = 50,ncol = 5)
y_data <-  ifelse(runif(50,0,1) > 0.6, 1L,0L) %>% as.matrix()

# データ作成
# --- 検証データ
x_data2 <- matrix(data = runif(500,0,1),nrow = 50,ncol = 5)
y_data2 <-  ifelse(runif(50,0,1) > 0.6, 1L,0L) %>% as.matrix()


# 2 モデル構築 -----------------------------------------------------------------------------

# モデル構築
build_model = function(hp) {

  model = keras_model_sequential()
  model %>% layer_dense(units = hp$Int('units',
                                     min_value = 32,
                                     max_value = 512,
                                     step=  32),input_shape = ncol(x_data),
                        activation =  'relu') %>%
    layer_dense(units = 1, activation = 'softmax') %>%
    compile(
      optimizer = tf$keras$optimizers$Adam(
        hp$Choice('learning_rate',
                  values=c(1e-2, 1e-3, 1e-4))),
      loss = 'binary_crossentropy',
      metrics = 'accuracy')
  return(model)
}


# 3 チューニング --------------------------------------------------------------------------------

# Tuner作成
# --- インスタンス化
tuner = RandomSearch(
    build_model,
    objective = 'val_accuracy',
    max_trials = 5,
    executions_per_trial = 3,
    directory = 'my_dir',
    project_name = 'helloworld')

tuner %>% search_summary()

tuner %>% fit_tuner(x_data,y_data,
                    epochs = 5,
                    validation_data = list(x_data2,y_data2))


# 4 結果検証 -------------------------------------------------------------------------

# チューナーの表示
# ---
result = tuner %>% plot_tuner(height = 500, width = 500)
result
# the list will show the plot and the data.frame of tuning results

# サマリー表示
tuner %>% results_summary(num_trials = 5)

# 上位モデルの選択
best_5_models = tuner %>% get_best_models(num_models = 5)
best_5_models

# モデル構造の表示
best_5_models[[1]] %>%
  plot_keras_model(show_shapes = TRUE, dpi = 96)
