# ***************************************************************************************
# Title     : Hyperparameter Tuning
# Objective : TODO
# Created by: Owner
# Created on: 2020/11/18
# URL       : https://tensorflow.rstudio.com/tools/tfruns/tuning/
# ***************************************************************************************


# ＜ポイント＞
# - ディープラーニングではハイパーパラメータのチューニングが必要になる
# - {tfruns}はトレーニングスクリプトのソースコードを変更するのではなく実行することができる
#   --- キーパラメータのフラグを定義
#   --- それらのフラグの組み合わせをトレーニング
#   --- どのフラグの組み合わせが最適なモデルになるかを判断


# ＜目次＞
# 0 環境設定
# 1 チューニングの実行



# 0 環境設定 ----------------------------------------------------------------------------

# ライブラリ
library(reticulate)
library(tidyverse)
library(keras)
library(tensorflow)
library(tfruns)


# 仮想環境の選択
use_condaenv("C:/Users/Owner/Anaconda3/envs/r-reticulate", required = TRUE)
py_config()




# 1 チューニングの実行 ----------------------------------------------------------------------------

# ＜ポイント＞
# - mnist_mlp.Rでは、チューニング個所をパラメータとして変更できるように設計している


# チューニング開始
# --- カレントディレクトリの｢runs｣フォルダにログが蓄積される
# --- 別のディレクトリを指定したい場合は｢runs_dir｣引数で指定
runs <- tuning_run("library/tfruns/tutorial/script/mnist_mlp.R",
                   flags = list(dropout1 = c(0.2, 0.3, 0.4),
                                dropout2 = c(0.2, 0.3, 0.4)))


# 実行結果の確認
# --- データフレーム形式で出力
runs %>% print()


# データ概要
runs %>% class()
runs %>% glimpse()



# 2 ランダムサンプリング ----------------------------------------------------------------------------

# ランダムサンプリング
# --- チューニングパターンが多い場合はsample引数でランダム抽出が可能
runs <- tuning_run("library/tfruns/tutorial/script/mnist_mlp.R",
                   sample = 0.3,
                   flags = list(dropout1 = c(0.2, 0.3, 0.4),
                                dropout2 = c(0.2, 0.3, 0.4)))


