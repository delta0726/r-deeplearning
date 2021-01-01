# ***************************************************************************************
# Title     : Quick Start
# Objective : TODO
# Created by: Owner
# Created on: 2020/11/1
# URL       : https://tensorflow.rstudio.com/tutorials/beginners/basic-ml/tutorial_overfit_underfit/
# ***************************************************************************************


# ＜オーバーフィッティングの回避策＞
# - 過剰適合を防ぐための最善の解決策は、より多くのトレーニングデータを使用すること
#   --- より多くのデータでトレーニングされたモデルはより一般化される
# - 正則化などの手法を使用する
#   --- モデルが保存できる情報の量とタイプに制約を課す
#   --- ネットワークが少数のパターンしか記憶できない場合、最適化プロセスにより、最も顕著なパターンに焦点を合わせるように強制される


# ＜目次＞
# 0. 準備
# 1. データ準備
# 2. 過剰適合とは
# 3. モデル構築(ベースラインモデル)
# 4. モデル構築(小規模モデル)
# 5. モデル構築(大規模モデル)
# 6. 結果比較
# 7. 正則化の追加
# 8. ドロップアウトの追加



# 0. 準備 -------------------------------------------------------------------------------

# ライブラリ
library(reticulate)
library(tidyverse)
library(keras)


# 仮想環境の選択
use_condaenv("C:/Users/Owner/Anaconda3/envs/r-reticulate", required = TRUE)
py_config()



# 1. データ準備 -------------------------------------------------------------------------------


# データ準備
# --- IMDBデータ
# --- データ量を限定することで過剰適合の発生を確認
num_words <- 1000
imdb <- dataset_imdb(num_words = num_words)


# データ分割
c(train_data, train_labels) %<-% imdb$train
c(test_data, test_labels) %<-% imdb$test


# 関数定義
# --- マルチホットエンコーディング
# --- リストを0と1のベクトルに変換する
multi_hot_sequences <- function(sequences, dimension) {
  multi_hot <- matrix(0, nrow = length(sequences), ncol = dimension)
  for (i in 1:length(sequences)) {
    multi_hot[i, sequences[[i]]] <- 1
  }
  multi_hot
}


# マルチホットエンコーディング
train_data <- train_data %>% multi_hot_sequences(num_words)
test_data  <- test_data %>% multi_hot_sequences(num_words)


# プロット用データ
# --- 単語インデックスは頻度でソートされている
first_text <- data.frame(word = 1:num_words, value = train_data[1, ])


# プロット
# --- インデックスゼロの近くに1つの値が多い
first_text %>%
  ggplot(aes(x = word, y = value)) +
    geom_line() +
    theme(axis.title.y = element_blank(),
          axis.text.y = element_blank(),
          axis.ticks.y = element_blank())



# 2. 過剰適合とは --------------------------------------------------------------------------

# ＜ポイント＞
# - 剰適合を防ぐ最も簡単な方法は、モデルのサイズ(学習可能なパラメーターの数)を減らすことです。
#   --- レイヤーの数とレイヤーあたりのユニットの数によって決定される
#   --- 学習可能なパラメーターの数は、モデルの｢容量｣と呼ばれることがよくある
# - パラメータが多いモデルほど｢記憶能力｣が高くなるため、トレーニングサンプルとそのターゲット間の完全なマッピングを行う
#   --- 汎化能力のないマッピングが行われ、予測能力が期待できない
#   --- ディープラーニングの目的は｢適合｣ではなく｢一般化｣
# - 一方、ネットワークの記憶リソースが限られている場合、マッピングを簡単に学習することはできません（学習不足）
#   --- 損失を最小限に抑えるには、より予測力のある圧縮表現を学習する必要がある


# ＜方針＞
# - モデルの適切なサイズまたはアーキテクチャを決定するための魔法の公式はない
# - 比較的少数のレイヤーとパラメーターから始めて、検証損失の収穫逓減が見られるまでモデルを拡張していく
#   --- レイヤーのサイズを増やす
#   --- 新しいレイヤーを追加する



# 3. モデル構築(ベースラインモデル) --------------------------------------------------------------------

# ネットワーク構築
baseline_model <-
  keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = num_words) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")


# コンパイル
baseline_model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = list("accuracy")
)


# サマリー
baseline_model %>% summary()


# トレーニング実行
baseline_history <-
  baseline_model %>%
    fit(train_data,
        train_labels,
        epochs = 20,
        batch_size = 512,
        validation_data = list(test_data, test_labels),
        verbose = 2)


# 4. モデル構築(小規模モデル) -------------------------------------------------------------------

# ネットワーク構築
smaller_model <-
  keras_model_sequential() %>%
    layer_dense(units = 4, activation = "relu", input_shape = num_words) %>%
    layer_dense(units = 4, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")


# コンパイル
smaller_model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = list("accuracy")
)


# サマリー
smaller_model %>% summary()


# トレーニング実行
smaller_history <-
  smaller_model %>%
    fit(train_data,
        train_labels,
        epochs = 20,
        batch_size = 512,
        validation_data = list(test_data, test_labels),
        verbose = 2)



# 5. モデル構築(大規模モデル) -------------------------------------------------------------------

# ネットワーク構築
bigger_model <-
  keras_model_sequential() %>%
  layer_dense(units = 512, activation = "relu", input_shape = num_words) %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")


# コンパイル
bigger_model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = list("accuracy")
)


# サマリー
bigger_model %>% summary()


# トレーニング実行
bigger_history <-
  bigger_model %>%
    fit(train_data,
        train_labels,
        epochs = 20,
        batch_size = 512,
        validation_data = list(test_data, test_labels),
        verbose = 2)


# 6. 結果比較 -------------------------------------------------------------------------------

# データ作成
# --- プロット用
compare_cx <-
  data.frame(baseline_train = baseline_history$metrics$loss,
             baseline_val   = baseline_history$metrics$val_loss,
             smaller_train  = smaller_history$metrics$loss,
             smaller_val    = smaller_history$metrics$val_loss,
             bigger_train   = bigger_history$metrics$loss,
             bigger_val     = bigger_history$metrics$val_loss) %>%
  rownames_to_column() %>%
  mutate(rowname = as.integer(rowname)) %>%
  gather(key = "type", value = "value", -rowname)


# プロット作成
# --- 結果比較
compare_cx %>%
  ggplot(aes(x = rowname, y = value, color = type)) +
    geom_line() +
    xlab("epoch") +
    ylab("loss")



# 7. 正則化の追加 -------------------------------------------------------------------------------

# ＜ポイント＞
# - 2つの説明が与えられた場合、正しいと思われる説明は｢最も単純なもの｣｢仮定の量が最も少ないもの｣である
#   --- DLにおける｢単純なモデル｣とは、パラメーター値の分布のエントロピーが少ないモデル
# - 過剰適合を軽減する方法の1つは、ネットワークの重みに小さな値のみを適用することでネットワークの複雑さに制約を課すこと（正則化）
#   --- これにより、重み値の分布がより｢規則的｣になる
#   --- ネットワークの損失関数に大きな重みを持つことに関連するコストを追加することによって行われます
# - 正則化には｢L1正則化｣と｢L2正則化｣の2種類がある
#   --- L1正則化： 追加されるコストは、ウエイト係数の絶対値に比例（Lasso）
#   --- L2正則化： 追加されるコストは、ウエイト係数の二乗に比例（Ridge）


# モデル構築
# --- レイヤーに正則化を追加
l2_model <-
  keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = num_words,
              kernel_regularizer = regularizer_l2(l = 0.001)) %>%
  layer_dense(units = 16, activation = "relu",
              kernel_regularizer = regularizer_l2(l = 0.001)) %>%
  layer_dense(units = 1, activation = "sigmoid")


# コンパイル
l2_model %>%
  compile(optimizer = "adam",
          loss = "binary_crossentropy",
          metrics = list("accuracy"))


# サマリー
l2_model %>% summary()


# トレーニング実行
l2_history <-
  l2_model %>%
    fit(train_data,
        train_labels,
        epochs = 20,
        batch_size = 512,
        validation_data = list(test_data, test_labels),
        verbose = 2)



# データ作成
# --- プロット用
compare_cx <-
  data.frame(baseline_train = baseline_history$metrics$loss,
             baseline_val = baseline_history$metrics$val_loss,
             l2_train = l2_history$metrics$loss,
             l2_val = l2_history$metrics$val_loss) %>%
  rownames_to_column() %>%
  mutate(rowname = as.integer(rowname)) %>%
  gather(key = "type", value = "value", -rowname)


# プロット作成
# --- L2正則化が入るとLossが過剰に減少しなくなる
# --- 過剰適合に対して耐性を得ている
compare_cx %>%
  ggplot(aes(x = rowname, y = value, color = type)) +
    geom_line() +
    xlab("epoch") +
    ylab("loss")




# 8. ドロップアウトの追加 ------------------------------------------------------------------------

# ＜ポイント＞
# - ニューラルネットワークで最も効果的で最も一般的に使用されている正則化手法の1つ
# - ドロップアウトは、トレーニング中にレイヤーのいくつかの出力機能をランダムに｢ドロップアウト｣(ゼロに設定)することで構成


# モデル構築
# --- ドロップアウトを追加
dropout_model <-
  keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = num_words) %>%
  layer_dropout(0.6) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dropout(0.6) %>%
  layer_dense(units = 1, activation = "sigmoid")


# コンパイル
dropout_model %>%
  compile(optimizer = "adam",
          loss = "binary_crossentropy",
          metrics = list("accuracy"))


# サマリー
# --- 設定は表示されない
dropout_model %>% summary()


# トレーニング実行
dropout_history <-
  dropout_model %>%
    fit(train_data,
        train_labels,
        epochs = 20,
        batch_size = 512,
        validation_data = list(test_data, test_labels),
        verbose = 2)


# データ作成
# --- プロット用
compare_cx <-
  data.frame(baseline_train = baseline_history$metrics$loss,
             baseline_val   = baseline_history$metrics$val_loss,
             dropout_train  = dropout_history$metrics$loss,
             dropout_val    = dropout_history$metrics$val_loss) %>%
  rownames_to_column() %>%
  mutate(rowname = as.integer(rowname)) %>%
  gather(key = "type", value = "value", -rowname)


# プロット作成
# --- L2正則化が入るとLossが過剰に減少しなくなる
# --- 過剰適合に対して耐性を得ている
compare_cx %>%
  ggplot(aes(x = rowname, y = value, color = type)) +
  geom_line() +
  xlab("epoch") +
  ylab("loss")

