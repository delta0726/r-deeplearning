# ***************************************************************************************
# Title     : tfruns: Track and Visualize Training Runs
# Objective : TODO
# Created by: Owner
# Created on: 2020/11/18
# URL       : https://tensorflow.rstudio.com/tools/tfruns/overview/
# ***************************************************************************************


# ＜tfrunsとは＞
# - 主に以下の3つの機能を持つ
# - 1. 実行結果のログ管理/閲覧
# - 2. ハイパーパラメータのチューニング
# - 3. tensorflow::tensorboard()との連携


# ＜ポイント＞
# - ソースコードを変更することなくハイパーパラメータのチューニングを行う
# - 実行結果はデフォルトでカレントディレクトリに｢runs｣フォルダを作成して格納される
# --- ｢I:\Project\R\deeplearning\runs｣にログが格納されている


# ＜その他＞
# - {tfruns}はR独自のパッケージのようだ

# ＜目次＞
# 0 環境設定
# 1 スクリプト実行
# 2 ログの確認
# 3 実行結果の分析



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



# 1 スクリプト実行 ----------------------------------------------------------------------------

# カレントディレクトリの確認
run_dir()


# スクリプトの実行
# --- 実行に加えてログ管理を行う
# --- TF2.0ではWarningが出るが、実行はできている模様
training_run("library/tfruns/tutorial/script/mnist_mlp.R")



# 2 ログの確認 ----------------------------------------------------------------------------

# 結果確認
# --- 直近の実行結果
# --- ディレクトリ名がタイムスタンプになっている
latest_run()


# 実行結果を指定して確認
view_run("runs/2020-11-18T00-54-56Z")


run_info()



# 3 実行結果の分析 ----------------------------------------------------------------------------

# 結果一覧の表示
# --- データフレーム形式
run_info(run_dir = "runs")


# 抽出条件を指定
ls_runs(eval_accuracy > 0.98, order = eval_accuracy)

