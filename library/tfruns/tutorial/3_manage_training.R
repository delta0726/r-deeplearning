# ***************************************************************************************
# Title     : Managing Runs
# Objective : TODO
# Created by: Owner
# Created on: 2020/11/18
# URL       : https://tensorflow.rstudio.com/tools/tfruns/managing/
# ***************************************************************************************



# ＜ポイント＞
# - 結果管理のユーティリティ関数が提供されている




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




# 1 ファイル管理 ----------------------------------------------------------------------------

# 実行結果のコピー
copy_run_files("2020-11-18T01-37-59Z", to = "library/tfruns/tutorial/copy/")


# アーカイブ
# --- データが不要になった一連の実行をアーカイブ
clean_runs(ls_runs(eval_acc < 0.98))


purge_runs(runs_dir = "library/tfruns/tutorial/copy/")