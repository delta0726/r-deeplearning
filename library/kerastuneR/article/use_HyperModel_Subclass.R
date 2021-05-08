# ***************************************************************************************
# Library   : kerastuneR
# Title     : Use HyperModel Subclass
# Created by: Owner
# Created on: 2021/05/07
# URL       : https://github.com/EagerAI/kerastuneR
# ***************************************************************************************


# ＜概要＞
# - {reticulate}でRからPythonのクラスを定義してHyperModelサブクラスを作成する


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

# データ作成
# --- 訓練データ
x_data <- matrix(data = runif(500,0,1),nrow = 50,ncol = 5)
y_data <- ifelse(runif(50,0,1) > 0.6, 1L,0L) %>% as.matrix()

# データ作成
# --- 検証データ
x_data2 <- matrix(data = runif(500,0,1),nrow = 50,ncol = 5)
y_data2 <- ifelse(runif(50,0,1) > 0.6, 1L,0L) %>% as.matrix()


# 2 モデル構築 ----------------------------------------------------------------------------

HyperModel <- reticulate::PyClass(
  'HyperModel',
  inherit = kerastuneR::HyperModel_class(),
  list(

    `__init__` = function(self, num_classes) {

      self$num_classes = num_classes
      NULL
    },
    build = function(self,hp) {
      model = keras_model_sequential()
      model %>% layer_dense(units = hp$Int('units',
                                           min_value = 32,
                                           max_value = 512,
                                           step = 32),
                            input_shape = ncol(x_data),
                            activation = 'relu') %>%
        layer_dense(as.integer(self$num_classes), activation = 'softmax') %>%
        compile(
          optimizer = tf$keras$optimizers$Adam(
            hp$Choice('learning_rate',
                      values = c(1e-2, 1e-3, 1e-4))),
          loss = 'sparse_categorical_crossentropy',
          metrics = 'accuracy')
    }
  )
)

hypermodel = HyperModel(num_classes = 10)


tuner = RandomSearch(hypermodel = hypermodel,
                      objective = 'val_accuracy',
                      max_trials = 2,
                      executions_per_trial = 1,
                      directory = 'my_dir5',
                      project_name = 'helloworld')

