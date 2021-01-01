# ***************************************************************************************
# Title     : Save and Restore Models
# Objective : TODO
# Created by: Owner
# Created on: 2020/11/20
# URL       : https://tensorflow.rstudio.com/tutorials/beginners/basic-ml/tutorial_save_and_restore/
# ***************************************************************************************


# ＜ポイント＞
# - モデルは、トレーニング後およびトレーニング中に保存できます
#    --- モデルが中断したところから再開でき、長いトレーニング時間を回避できることを意味する
#
#


# ＜目次＞
# 0 環境準備
# 1 データ準備
# 2 モデル構築
# 3 SavedModelフォーマットで保存
# 4 HDF5フォーマットで保存
# 5 Checkpoint コールバック
# 6 コールバックのオプション設定
# 7. 手動でのウエイト保存



# 0 環境準備 -------------------------------------------------------------------------------

# ライブラリ
library(reticulate)
library(magrittr)
library(tidyverse)
library(keras)


# 仮想環境の選択
use_condaenv("C:/Users/Owner/Anaconda3/envs/r-reticulate", required = TRUE)
py_config()



# 1 データ準備 -------------------------------------------------------------------------------

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



# 2 モデル構築 -------------------------------------------------------------------------------

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



# 3 SavedModelフォーマットで保存 ---------------------------------------------------------------

# ＜ポイント＞
# - モデルをシリアル化して保存
#   --- save_model_tf()で保存して、load_model_tf()で復元する

# モデル構築
model <- create_model()


# トレーニング実行
model %>% fit(train_images, train_labels, epochs = 5, verbose = 2)


# モデル保存
model %>% save_model_tf("library/tensorflow/tutorials/model")


# 保存先の確認
# --- フォルダとファイルに分けて保存されている
# --- 1つのファイルになっているわけではない
list.files("library/tensorflow/tutorials/model")


# モデル復元
new_model <- load_model_tf("library/tensorflow/tutorials/model")


# 比較
model %>% summary()
new_model %>% summary()


# モデル削除
rm(model, new_model)


# 4. HDF5フォーマットで保存 ---------------------------------------------------------------

# ＜ポイント＞
# - HDF5フォーマットでは｢ウエイト｣｢モデル構成｣｢オプティマイザ｣の全てが保存される
#


# モデル構築
model <- create_model()


# トレーニング実行
model %>% fit(train_images, train_labels, epochs = 5, verbose = 2)


# モデル保存
# --- ディレクトリを予め作っていないとエラーになる
model %>% save_model_hdf5("library/tensorflow/tutorials/model_h5/my_model.h5")


# 保存先の確認
list.files("library/tensorflow/tutorials/model_h5")


# モデル復元
new_model <- load_model_hdf5("library/tensorflow/tutorials/model_h5/my_model.h5")


# 比較
model %>% summary()
new_model %>% summary()


# モデル削除
rm(model, new_model)



# 5 Checkpoint コールバック ---------------------------------------------------------------

# ＜ポイント＞
# - トレーニング中およびトレーニングの終了時にチェックポイントを自動的に保存すると便利
#   --- トレーニングされたモデルを再トレーニングすることなく使用
#   --- トレーニングプロセスが中断された場合に備えて、離れた場所でトレーニングをピックアップすることが可能


# チェックポイントの保存先
checkpoint_path <- "library/tensorflow/tutorials/checkpoints/cp.ckpt"


# コールバック関数
# --- チェックポイントの作成
cp_callback <- callback_model_checkpoint(
  filepath = checkpoint_path,
  save_weights_only = TRUE,
  verbose = 0
)


# モデル作成
model <- create_model()


# 学習
# --- コールバックでチェックポイントを保存
model %>%
  fit(train_images, train_labels,
      epochs = 10,
      validation_data = list(test_images, test_labels),
      callbacks = list(cp_callback),  # pass callback to training
      verbose = 2)


# 保存先の確認
list.files(dirname(checkpoint_path))


# 新モデルの構築
# --- 未トレーニングの状態でモデル評価
# --- Accuracy=0.098
fresh_model <- create_model()
fresh_model %>% evaluate(test_images, test_labels, verbose = 0)


# モデルのロード
# --- チェックポイントのロード
# --- Accuracy=0.879
fresh_model %>% load_model_weights_tf(filepath = checkpoint_path)
fresh_model %>% evaluate(test_images, test_labels, verbose = 0)



# 6 コールバックのオプション設定 -----------------------------------------------------------------

# チェックポイントの保存先
checkpoint2_path <- "library/tensorflow/tutorials/checkpoints2/cp.ckpt"


# コールバック関数
# --- チェックポイントの作成
# --- save_best_only = TRUE
cp_callback <-
  callback_model_checkpoint(filepath = checkpoint2_path,
                            save_weights_only = TRUE,
                            save_best_only = TRUE,
                            verbose = 1)


# 新モデルの構築
model <- create_model()


model %>%
  fit(train_images, train_labels,
      epochs = 10,
      validation_data = list(test_images, test_labels),
      callbacks = list(cp_callback),
      verbose = 2)


# 保存先の確認
list.files(dirname(checkpoint2_path))




# 7. 手動でのウエイト保存 -----------------------------------------------------------------

# ウエイト保存
# --- 直前モデル
model %>% save_model_weights_tf("library/tensorflow/tutorials/weight/cp.ckpt")


# モデル作成
new_model <- create_model()


# ウエイトのロード
new_model %>% load_model_weights_tf('library/tensorflow/tutorials/weight/cp.ckpt')


# モデル評価
new_model %>% evaluate(test_images, test_labels, verbose = 0)


