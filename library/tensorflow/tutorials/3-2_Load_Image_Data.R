# ***************************************************************************************
# Title     : Save and Restore Models
# Objective : TODO
# Created by: Owner
# Created on: 2020/11/2
# URL       : https://tensorflow.rstudio.com/tutorials/beginners/basic-ml/tutorial_save_and_restore/
# ***************************************************************************************


# ＜オーバーフィッティングの回避策＞
# - 過剰適合を防ぐための最善の解決策は、より多くのトレーニングデータを使用すること
#   --- より多くのデータでトレーニングされたモデルはより一般化される
# - 正則化などの手法を使用する
#   --- モデルが保存できる情報の量とタイプに制約を課す
#   --- ネットワークが少数のパターンしか記憶できない場合、最適化プロセスにより、最も顕著なパターンに焦点を合わせるように強制される


# ＜目次＞
# 0. 準備
# 1. モデル構築
# 2. モデルトレーニング
# 3. SavedModelフォーマットで保存
# 4. HDF5フォーマットで保存
# 5. Checkpoint コールバック



# 0. 準備 -------------------------------------------------------------------------------

# 環境設定
Sys.setenv(RETICULATE_PYTHON = "C:/Users/Owner/Anaconda3/envs/r-reticulate")


# ライブラリ
library(reticulate)
library(magrittr)
library(tidyverse)
library(keras)


# 仮想環境の選択
use_condaenv("r-reticulate", required = TRUE)
py_config()


# データ準備
# --- MNISTデータ
mnist <- dataset_mnist()


# データ分割
c(train_images, train_labels) %<-% mnist$train
c(test_images, test_labels) %<-% mnist$test


# ラベル取得
# --- 1000個のみ
train_labels <- train_labels[1:1000]
test_labels <- test_labels[1:1000]


# データ変換
train_images <- train_images[1:1000, , ] %>% array_reshape(c(1000, 28 * 28)) %>% divide_by(255)
test_images  <- test_images[1:1000, , ] %>% array_reshape(c(1000, 28 * 28)) %>% divide_by(255)



# 1. モデル構築 -------------------------------------------------------------------------------

# 関数定義
# --- ネットワーク構築
# --- コンパイル
create_model <- function() {

  # ネットワーク構築
  model <-
    keras_model_sequential() %>%
      layer_dense(units = 512, activation = "relu", input_shape = 784) %>%
      layer_dropout(0.2) %>%
      layer_dense(units = 10, activation = "softmax")

  # コンパイル
  model %>%
    compile(optimizer = "adam",
            loss = "sparse_categorical_crossentropy",
            metrics = list("accuracy"))

  # アウトプット
  model
}


# モデル構築
model <- create_model()


# サマリー
model %>% summary()



# 2. モデルトレーニング ------------------------------------------------------------------------

# トレーニング実行
model %>% fit(train_images, train_labels, epochs = 5, verbose = 2)




# 3. SavedModelフォーマットで保存 ---------------------------------------------------------------

# ＜ポイント＞
# - モデルをシリアル化して保存
#   --- save_model_tf()で保存して、load_model_tf()で復元する


# モデル保存
model %>% save_model_tf("library/tensorflow/tutorials/model")


# モデル復元
new_model <- load_model_tf("library/tensorflow/tutorials/model")


# 比較
model %>% summary()
new_model %>% summary()



# 4. HDF5フォーマットで保存 ---------------------------------------------------------------

# ＜ポイント＞
# - HDF5フォーマットでは｢ウエイト｣｢モデル構成｣｢オプティマイザ｣の全てが保存される
#

# モデル保存
# --- ディレクトリを予め作っていないとエラーになる
model %>% save_model_hdf5("library/tensorflow/tutorials/model_h5/my_model.h5")


# モデル復元
new_model <- load_model_hdf5("library/tensorflow/tutorials/model_h5/my_model.h5")
new_model %>% summary()


# 比較
model %>% summary()
new_model %>% summary()



# 5. Checkpoint コールバック ---------------------------------------------------------------

checkpoint_path <- "checkpoints/cp.ckpt"

# Create checkpoint callback
cp_callback <- callback_model_checkpoint(
  filepath = checkpoint_path,
  save_weights_only = TRUE,
  verbose = 0
)

model <- create_model()

model %>% fit(
  train_images,
  train_labels,
  epochs = 10,
  validation_data = list(test_images, test_labels),
  callbacks = list(cp_callback),  # pass callback to training
  verbose = 2
)


list.files(dirname(checkpoint_path))


fresh_model <- create_model()
fresh_model %>% evaluate(test_images, test_labels, verbose = 0)



fresh_model %>% load_model_weights_tf(filepath = checkpoint_path)
fresh_model %>% evaluate(test_images, test_labels, verbose = 0)



# 6. Checkpoint コールバック -----------------------------------------------------------------


checkpoint_path <- "checkpoints/cp.ckpt"

# Create checkpoint callback
cp_callback <- callback_model_checkpoint(
  filepath = checkpoint_path,
  save_weights_only = TRUE,
  save_best_only = TRUE,
  verbose = 1
)

model <- create_model()

model %>% fit(
  train_images,
  train_labels,
  epochs = 10,
  validation_data = list(test_images, test_labels),
  callbacks = list(cp_callback), # pass callback to training,
  verbose = 2
)


list.files(dirname(checkpoint_path))




# 7. 手動でのウエイト保存 -----------------------------------------------------------------

# Save the weights
model %>% save_model_weights_tf("checkpoints/cp.ckpt")

# Create a new model instance
new_model <- create_model()

# Restore the weights
new_model %>% load_model_weights_tf('checkpoints/cp.ckpt')

# Evaluate the model
new_model %>% evaluate(test_images, test_labels, verbose = 0)


