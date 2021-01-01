# ***************************************************************************************
# Title     : Text Classification
# Objective : TODO
# Created by: Owner
# Created on: 2020/11/19
# URL       : https://tensorflow.rstudio.com/tutorials/beginners/basic-ml/tutorial_basic_text_classification/
# ***************************************************************************************


# ＜目的＞
# - レビューのテキストを使用して映画のレビューをポジティブまたはネガティブに分類する
#   --- これはバイナリ分類の例であり、重要で広く適用可能な種類の機械学習の問題の事例となる


# ＜ポイント＞
# - 1. 画像を分類するニューラルネットワークを構築します。
# - 2. ニューラルネットワークのトレーニングします。
# - 3. モデルの精度評価。
# - 4. 作成したモデルを保存して復元します。


# ＜目次＞
# 0. 準備
# 1. データ確認
# 2. データ分割
# 3. テキストのベクトル化
# 4. モデル構築
# 5. モデルトレーニング
# 6. モデル評価



# 0. 準備 -------------------------------------------------------------------------------

# ライブラリ
library(reticulate)
library(tidyverse)
library(keras)
library(pins)


# 仮想環境の選択
use_condaenv("C:/Users/Owner/Anaconda3/envs/r-reticulate", required = TRUE)
py_config()


# データのダウンロード
# --- IMDBデータセット
# --- トレーニング用の25,000件のレビューとテスト用の25,000件のレビュー
board_register_kaggle(token = "C:/kaggle/kaggle.json")
paths <- pins::pin_get("nltkdata/movie-review", "kaggle")


# データパスの取得
path <- paths[1]


# データ取得
# --- 映画レビューのテキスト
#--- 映画のレビューをポジティブまたはネガティブに分類
df <- readr::read_csv(path)



# 1. データ確認 -------------------------------------------------------------------------------

# データ確認
# --- ほぼ半数がPositive/Negative
df %>% head()
df %>% count(tag)


# レビューのイメージ
df$text[1]



# 2. データ分割 -------------------------------------------------------------------------------

# データ分割
training_id <- df %>% nrow() %>% sample.int(size = nrow(df)*0.8)
training <- df[training_id,]
testing <- df[-training_id,]



# 3. テキストのベクトル化 -------------------------------------------------------------------------------

# ＜ポイント＞
# - レビュー(テキスト)は、ニューラルネットワークに入力する前にテンソルに変換する必要がある


# レビューごとの長さを確認
df$text %>%
  strsplit(" ") %>%
  sapply(length) %>%
  summary()


# テキストのベクトル化
num_words <- 10000
max_length <- 50
text_vectorization <-
  layer_text_vectorization(max_tokens = num_words,
                           output_sequence_length = max_length)


text_vectorization %>%
  adapt(df$text)


# ボキャブラリーの取得
# TODO see https://github.com/tensorflow/tensorflow/pull/34529
text_vectorization %>% get_vocabulary()



# 4. モデル構築 -------------------------------------------------------------------------------

# ＜ポイント＞
# - ニューラルネットワークは、レイヤーを積み重ねることによって作成される
# - 2つの主要なアーキテクチャ上の決定が必要
#   --- モデルで使用するレイヤーはいくつ
#   --- 各レイヤーに使用する非表示のユニットはいくつ


# 入力層の設定
input <- layer_input(shape = c(1), dtype = "string")


# 出力層の設定
# --- layer_embedding： 整数でエンコードされた語彙を取得し、各単語インデックスの埋め込みベクトルを検索する
# --- global_average_pooling_1d : シーケンスディメンションを平均することにより、各例の固定長の出力ベクトルを返す
output <-
  input %>%
    text_vectorization() %>%
    layer_embedding(input_dim = num_words + 1, output_dim = 16) %>%
    layer_global_average_pooling_1d() %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dropout(0.5) %>%
    layer_dense(units = 1, activation = "sigmoid")


# モデル構築
model <- keras_model(input, output)


# モデルのコンパイル
# --- オプティマイザと損失関数を設定
# --- binary_crossentropyは確率を処理するのに適している
model %>%
  compile(optimizer = 'adam',
          loss = 'binary_crossentropy',
          metrics = list('accuracy'))



# 5. モデルトレーニング -------------------------------------------------------------------------------


# 機械学習／ディープラーニングにおけるバッチサイズ、イテレーション数、エポック数の決め方
# https://qiita.com/kenta1984/items/bad75a37d552510e4682

# モデル訓練
# --- batch_size： 学習の際にデータセットを分割するにあたり、1つのサブセットのデータ数
# --- epochs: サブセット全体の学習を1エポックとして、何セットの学習を行うか
# --- validation_split: サブセットの何％のデータを検証に使うか（シャッフルしたい場合はshuffleを指定）
history <-
  model %>%
    fit(training$text,
        as.numeric(training$tag == "pos"),
        epochs = 10,
        batch_size = 512,
        validation_split = 0.2,
        verbose = 2)


# 結果確認
history %>% print()
history %>% glimpse()


# プロット
history %>% plot()



# 6. モデル評価 -------------------------------------------------------------------------------

results <- model %>% evaluate(testing$text, as.numeric(testing$tag == "pos"), verbose = 0)
results


