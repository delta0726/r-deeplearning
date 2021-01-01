# ***************************************************************************************
# Title     : Regression
# Objective : TODO
# Created by: Owner
# Created on: 2020/11/18
# URL       : https://tensorflow.rstudio.com/tutorials/beginners/basic-ml/tutorial_basic_regression/
# ***************************************************************************************


# ＜ポイント＞
# - 回帰問題に使用される損失関数として平均二乗誤差(MSE)がある
#   --- 評価指標としても使用される
# - 特徴量はそれぞれスケーリングしておく必要がある
#   --- {tfdatasets}ではデータフレームを加工するレシピが準備されている
# - 過剰適合を防ぐための便利なテクニックとしてアーリーストッピングがある


# ＜目次＞
# 0 環境準備
# 1 データ準備
# 2 データ加工
# 3 データ前処理
# 4 ネットワーク構築
# 5 モデルのコンパイル
# 6 モデリングの関数化
# 7-1 モデルのトレーニング1
# 7-2 モデルのトレーニング2
# 8 メトリック出力
# 9 モデル予測



# 0 環境準備 -------------------------------------------------------------------------------

# ライブラリ
library(reticulate)
library(tidyverse)
library(keras)
library(tfdatasets)


# 仮想環境の選択
use_condaenv("C:/Users/Owner/Anaconda3/envs/r-reticulate", required = TRUE)
py_config()



# 1 データ準備 ----------------------------------------------------------------------------

# データ準備
# --- Boston Housing Prices dataset
boston_housing <- dataset_boston_housing()


# データ変換
c(train_data, train_labels) %<-% boston_housing$train
c(test_data, test_labels) %<-% boston_housing$test


# データ構造
train_data %>% glimpse()
train_labels %>% glimpse()
test_data %>% glimpse()
test_labels %>% glimpse()



# 2 データ加工 -------------------------------------------------------------------------------

# 列名の作成
column_names <-
  c('CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
    'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT')


# データ加工
# --- 訓練データ
train_df <-
  train_data %>%
    as_tibble(.name_repair = "minimal") %>%
    setNames(column_names) %>%
    mutate(label = train_labels)


# データ加工
# --- テストデータ
test_df <-
  test_data %>%
    as_tibble(.name_repair = "minimal") %>%
    setNames(column_names) %>%
    mutate(label = test_labels)


# データ確認
train_df %>% glimpse()
test_df %>% glimpse()



# 3 データ前処理 -------------------------------------------------------------------------------

# ＜ポイント＞
# - {tfdatasets}はfeature_spec()を通してデータ前処理を適用することができる
# - {recipes}と似ているが、実際にはそれほど充実したものではない
#    --- step_numeric_column()のように、データ型ごとにstep_*()が準備されている
#    --- recipesと同様にワークフローに融合して使用することができる


# 前処理
# --- 数値データを正規化
spec <-
  train_df %>%
    feature_spec(label ~ . ) %>%
      step_numeric_column(all_numeric(), normalizer_fn = scaler_standard()) %>%
      fit()


# 確認
spec %>% print()


# データ構造
spec %>% class()
spec %>% glimpse()


# 使い方
# --- specはlayer_dense_features()と一緒に使用する
# --- TensorFlowグラフの中で前処理を直接実行することができる
layer <-
  layer_dense_features(feature_columns = dense_features(spec),
                       dtype = tf$float32)

# データ確認
# ---pythonのtensorflowのクラスオブジェクト
train_df %>% layer()
train_df %>% layer() %>% class()



# 4 ネットワーク構築 -------------------------------------------------------------------------------

# インプット
# --- tfdatasets::layer_input_from_dataset()
input <-
  train_df %>%
    select(-label) %>%
    layer_input_from_dataset()


# 確認
# --- RのデータフレームがPythonの辞書型に変換されている
input %>% print()
input %>% class()


# ネットワーク構築
output <-
  input %>%
    layer_dense_features(dense_features(spec)) %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 1)


# 確認
# --- RのデータフレームがPythonの辞書型に変換されている
output %>% print()
output %>% class()


# モデル構築
model <- keras_model(input, output)


# 確認
model %>% summary()
model %>% glimpse()
model %>% class()



# 5 モデルのコンパイル -------------------------------------------------------------------------------

# コンパイル
# --- ｢損失関数｣｢オプティマイザー｣｢メトリック｣を指定
model %>%
  compile(
    loss = "mse",
    optimizer = optimizer_rmsprop(),
    metrics = list("mean_absolute_error")
  )



# 6 モデリングの関数化 -------------------------------------------------------------------------------

# 関数化
# --- さまざまな実験で再利用できるようにモデル構築プロセスを関数にまとめる
build_model <- function() {
  input <-
    train_df %>%
      select(-label) %>%
      layer_input_from_dataset()

  output <-
    input %>%
      layer_dense_features(dense_features(spec)) %>%
      layer_dense(units = 64, activation = "relu") %>%
      layer_dense(units = 64, activation = "relu") %>%
      layer_dense(units = 1)

  model <- keras_model(input, output)

  model %>%
    compile(
      loss = "mse",
      optimizer = optimizer_rmsprop(),
      metrics = list("mean_absolute_error")
    )

  model
}




# 7-1 モデルのトレーニング1 -------------------------------------------------------------------------------

# ＜ポイント＞
# - アーリーストッピングなしで学習


# コールバックの設定
# --- Display training progress by printing a single dot for each completed epoch.
print_dot_callback <- callback_lambda(
  on_epoch_end = function(epoch, logs) {
    if (epoch %% 80 == 0) cat("\n")
    cat(".")
  }
)


# モデル構築
model <- build_model()


# モデル訓練
history <-
  model %>%
    fit(x = train_df %>% select(-label),
        y = train_df$label,
        epochs = 500,
        validation_split = 0.2,
        verbose = 0,
        callbacks = list(print_dot_callback))


# プロット
# --- モデルのトレーニングの進行状況を視覚化
# --- 約200エポック後のモデルの改善がほとんどないことを示している
history %>% plot()



# 7-2 モデルのトレーニング2 -------------------------------------------------------------------------------

# ＜ポイント＞
# - アーリーストッピングを設定して学習


# モデル構築
model <- build_model()


# アーリーストッピングの設定
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20)


# モデル訓練
# --- アーリーストッピングあり
history <-
  model %>% fit(x = train_df %>% select(-label),
                y = train_df$label,
                epochs = 500,
                validation_split = 0.2,
                verbose = 0,
                callbacks = list(early_stop))


# プロット
# --- モデルのトレーニングの進行状況を視覚化
history %>% plot()



# 8 メトリック出力 ------------------------------------------------------------------------------

# メトリック出力
c(loss, mae) %<-% (model %>% evaluate(test_df %>% select(-label), test_df$label, verbose = 0))

paste0("Mean absolute error on test set: $", sprintf("%.2f", mae * 1000))




# 9 モデル予測 ------------------------------------------------------------------------------

# モデル予測
test_predictions <- model %>% predict(test_df %>% select(-label))
test_predictions[ , 1]


