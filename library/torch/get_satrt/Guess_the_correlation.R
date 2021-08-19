# ***************************************************************************************
# Library   : torch
# Title     : Guess the correlation
# Created by: Owner
# Created on: 2021/08/20
# URL       : https://torch.mlverse.org/start/guess_the_correlation/
# ***************************************************************************************


# ＜概要＞
# - {torch}を用いた基本的なワークフローを家訓する


# ＜目次＞
# 0 準備
# 1 データセットの取得
# 2 データの表示方法
# 3 プロット作成
# 4 モデル構築
# 5 ネットワークの学習
# 6 予測
# 7 プロット作成


# 0 準備 --------------------------------------------------------------------------------

# ライブラリ
library(tidyverse)
library(torch)
library(luz)
library(torchvision)
library(torchdatasets)


# 1 データセットの取得 ------------------------------------------------------------------

# サブセットの設定
train_indices <- 1:10000
val_indices <- 10001:15000
test_indices <- 15001:20000


# パラメータ設定
add_channel_dim <- function(img) img$unsqueeze(1)
crop_axes <- function(img) transform_crop(img, top = 0, left = 21, height = 131, width = 130)
root <- file.path(tempdir(), "correlation")

# データダウンロード
# --- 訓練データ
train_ds <- guess_the_correlation_dataset(
  root = root,
  transform = function(img) crop_axes(img) %>% add_channel_dim(),
  indexes = train_indices,
  download = TRUE
)

# データダウンロード
# --- 検証データ
valid_ds <- guess_the_correlation_dataset(
  root = root,
  transform = function(img) crop_axes(img) %>% add_channel_dim(),
  indexes = val_indices,
  download = FALSE
)

# データダウンロード
# --- テストデータ
test_ds <- guess_the_correlation_dataset(
  root = root,
  transform = function(img) crop_axes(img) %>% add_channel_dim(),
  indexes = test_indices,
  download = FALSE
)

# データ確認
train_ds %>% print()
train_ds %>% class()
train_ds %>% glimpse()
train_ds[1]

# データ量の確認 
train_ds %>% length()
valid_ds %>% length()
test_ds %>% length()


add_channel_dim <- function(img) img$unsqueeze(1)


crop_axes <- function(img) transform_crop(img, top = 0, left = 21, height = 131, width = 130)


# 2 データの表示方法 ---------------------------------------------------------------

# ＜ポイント＞
# - 一度にたくさんのデータを表示したいので、データのバッチを処理する方法を知る必要がある
# - 

# データローダーの作成
train_dl <- train_ds %>% dataloader(batch_size = 64, shuffle = TRUE)
valid_dl <- valid_ds %>% dataloader(batch_size = 64)
test_dl <- test_ds %>% dataloader(batch_size = 64)

# 確認
train_dl %>% print()
train_dl %>% class()

# データの長さ
# --- DataLoaderの場合はバッチの数を意味する
train_dl %>% length()
valid_dl %>% length()
test_dl %>% length()

# イテレータの作成
# --- バッチにアクセスするために使用
batch <- train_dl %>% dataloader_make_iter() %>% dataloader_next()

# 確認
batch %>% print()
batch %>% glimpse()

# 次元数
batch$x %>% dim()
batch$y %>% dim()


# 3 プロット作成 --------------------------------------------------------------------------

# プロット定義
par(mfrow = c(8,8), mar = rep(0, 4))

# イメージ取得
# --- 3次元データ
images <- batch$x$squeeze(2) %>% as.array()

# プロット作成
images %>%
  purrr::array_tree(1) %>%
  purrr::map(as.raster) %>%
  purrr::iwalk(~{plot(.x)})

# データアクセス
batch$y %>% as.numeric() %>% round(digits = 2)


# 4 モデル構築 -----------------------------------------------------------------------------

# ＜ポイント＞
# - 入力データが画像データなのでCNNを使用する
# - {torch}ではニューラルネットワークはモジュール
#   --- モデルを表すトップレベルモジュール
#   --- レイヤーを表すサブモジュール

# ＜構成要素＞
# initialize : サブモジュールをインスタンス化する場所
# forward    : このモジュールが呼び出されたときに何が起こるかを定義する場所


# initialize(イメージ用) --------------------------------------

# モデル構築
# --- レイヤーを定義（3つのconvレイヤー+2つの線形レイヤー）
# --- モデルの意味には本文より確認
net <- nn_module(

  initialize = function() {
    
    self$conv1 <- nn_conv2d(in_channels = 1, out_channels = 32, kernel_size = 3)
    self$conv2 <- nn_conv2d(in_channels = 32, out_channels = 64, kernel_size = 3)
    self$conv3 <- nn_conv2d(in_channels = 64, out_channels = 128, kernel_size = 3)
    
    self$fc1 <- nn_linear(in_features = 14 * 14 * 128, out_features = 128)
    self$fc2 <- nn_linear(in_features = 128, out_features = 1)
    
  },
  
)


# forward(イメージ用) ----------------------------------------

# モデル構築
# --- initializeで定義した層について処理を定義
net <- nn_module(

    forward = function(x) {
    
    x %>% 
      self$conv1() %>%
      nnf_relu() %>%
      nnf_avg_pool2d(2) %>%
      
      self$conv2() %>%
      nnf_relu() %>%
      nnf_avg_pool2d(2) %>%
      
      self$conv3() %>%
      nnf_relu() %>%
      nnf_avg_pool2d(2) %>%
      
      torch_flatten(start_dim = 2) %>%
      self$fc1() %>%
      nnf_relu() %>%
      
      self$fc2()
  }
)


# 本番用 ----------------------------------------------------

# シード設定
torch_manual_seed(777)


# モデル定義
net <- nn_module(
  
  "corr-cnn",
  
  initialize = function() {
    
    self$conv1 <- nn_conv2d(in_channels = 1, out_channels = 32, kernel_size = 3)
    self$conv2 <- nn_conv2d(in_channels = 32, out_channels = 64, kernel_size = 3)
    self$conv3 <- nn_conv2d(in_channels = 64, out_channels = 128, kernel_size = 3)
    
    self$fc1 <- nn_linear(in_features = 14 * 14 * 128, out_features = 128)
    self$fc2 <- nn_linear(in_features = 128, out_features = 1)
    
  },
  
  forward = function(x) {
    
    x %>% 
      self$conv1() %>%
      nnf_relu() %>%
      nnf_avg_pool2d(2) %>%
      
      self$conv2() %>%
      nnf_relu() %>%
      nnf_avg_pool2d(2) %>%
      
      self$conv3() %>%
      nnf_relu() %>%
      nnf_avg_pool2d(2) %>%
      
      torch_flatten(start_dim = 2) %>%
      self$fc1() %>%
      nnf_relu() %>%
      
      self$fc2()
  }
)


# 確認
# --- 学習前でもデータのバッチでモデルを呼び出すことが可能
# --- これにより、すべての形状が一致したかどうかがすぐに確認することができる
model <- net()
model(batch$x)


# 5 ネットワークの学習 ----------------------------------------------------------------------

# ＜ポイント＞
# - 学習は本来ループを伴う処理だが、{luz}を使用するとsetup()とfit()のみで処理される
#   --- setup()では｢損失関数｣と｢最適化アルゴリズム｣を定義する 
#   --- fit()では｢データローダー｣｢エポック数｣｢検証データ｣を与える

# 学習
# --- 学習は非常に遅い(20分くらいかかる)
# --- CPUもメモリも高稼働しているので正常に動作している
fitted <- 
  net %>%
    setup(loss = function(y_hat, y_true) nnf_mse_loss(y_hat, y_true$unsqueeze(2)),
          optimizer = optim_adam) %>%
  fit(train_dl, epochs = 10, valid_data = test_dl)



# 6 予測 -------------------------------------------------------------------------------------

# 予測値の取得
preds <- fitted %>% predict(test_dl)
preds <- preds$to(device = "cpu")$squeeze() %>% as.numeric()

# 実績値の取得
test_dl <- test_ds %>% dataloader(batch_size = 5000)
targets <- (test_dl %>% dataloader_make_iter() %>% dataloader_next())$y %>% as.numeric()


# 7 プロット作成 -----------------------------------------------------------------------------

# データ整理
df <- data.frame(preds = preds, targets = targets)

# プロット作成
df %>% 
  ggplot(aes(x = targets, y = preds)) +
    geom_point(size = 0.1) +
    theme_classic() +
    xlab("true correlations") +
    ylab("model predictions")
