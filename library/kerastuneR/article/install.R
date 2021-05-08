# ***************************************************************************************
# Library   : kerastuneR
# Title     : Install
# Created by: Owner
# Created on: 2021/05/07
# URL       : https://github.com/EagerAI/kerastuneR
# ***************************************************************************************


# ライブラリ
library(reticulate)
library(tidyverse)
library(keras)
library(tensorflow)
library(kerastuneR)

# 仮想環境の選択
use_condaenv("C:/Users/Owner/Anaconda3/envs/r-reticulate", required = TRUE)
py_config()

# バージョン確認
keras_tuner_version()

# インストール
install_kerastuner()

