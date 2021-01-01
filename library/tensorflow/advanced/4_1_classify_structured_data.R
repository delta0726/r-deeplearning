# ***************************************************************************************
# Title     : Classify structured data with feature columns
# Objective : TODO
# Created by: Owner
# Created on: 2020/10/
# URL       : https://tensorflow.rstudio.com/tutorials/advanced/structured/classify/
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



# 0 環境準備 -------------------------------------------------------------------------------

# ライブラリ
library(reticulate)
library(magrittr)
library(tidyverse)
library(keras)


# 仮想環境の選択
use_condaenv("C:/Users/Owner/Anaconda3/envs/r-reticulate", required = TRUE)
py_config()


