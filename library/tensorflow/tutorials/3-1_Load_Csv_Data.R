# ***************************************************************************************
# Title     : Load Image Data
# Objective : TODO
# Created by: Owner
# Created on: 2020/11/2
# URL       : https://tensorflow.rstudio.com/tutorials/beginners/load/load_image/
# ***************************************************************************************


# ＜目標＞
# - データのインポート方法を学ぶ
#   --- tfdataset::make_csv_dataset()
# - TensorFlowデータセットの特徴を学ぶ
# - {tfdataset}の前処理のパイプラインを学ぶ



# ＜目次＞
# 0 環境準備
# 1 データ準備
# 2 データ確認



# 0 環境準備 -------------------------------------------------------------------------------

# ライブラリ
library(reticulate)
library(magrittr)
library(tidyverse)
library(keras)
library(tfdatasets)


# 仮想環境の選択
use_condaenv("C:/Users/Owner/Anaconda3/envs/r-reticulate", required = TRUE)
py_config()




# 1 データ準備 -------------------------------------------------------------------------------

# URL指定
TRAIN_DATA_URL <- "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL <- "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"


# ファイル保存先
dir_file <- "library/tensorflow/tutorials/csv2"


# ファイル保存
train_file_path <- get_file(str_c(dir_file, "train.csv", sep = "/"), TRAIN_DATA_URL)
test_file_path <- get_file(str_c(dir_file, "evel.csv", sep = "/"), TEST_DATA_URL)


# データ取り込み
# --- 訓練データ
train_dataset <-
  train_file_path %>%
    make_csv_dataset(field_delim = ",",
                     batch_size = 5,
                     num_epochs = 1)


# データ取り込み
# --- テストデータ
test_dataset <-
  test_file_path %>%
    make_csv_dataset(field_delim = ",",
                     batch_size = 5,
                     num_epochs = 1)


# 2 データ確認 -------------------------------------------------------------------------------

# ＜ポイント＞
# - それぞれが列を表すテンソルのリストを作成している
# - Rのデータフレームと似ているが、大きな違いはTensorFlowデータセットがイテレータであるということ
#   --- iter_next()を呼び出すたびに、データセットから異なる行のバッチが生成される



train_dataset %>%
  reticulate::as_iterator() %>%
  reticulate::iter_next() %>%
  reticulate::py_to_r()


