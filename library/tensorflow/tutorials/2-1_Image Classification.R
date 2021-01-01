# ***************************************************************************************
# Title     : Basic Image Classification
# Objective : TODO
# Created by: Owner
# Created on: 2020/11/18
# URL       : https://tensorflow.rstudio.com/tutorials/beginners/basic-ml/tutorial_basic_classification/
# ***************************************************************************************


# ＜ポイント＞
# - ディープラーニングの基本タスクである｢Fashion MNIST｣を用いたチュートリアル
# - 画像の分類タスクの流れを確認


# ＜目次＞
# 0 環境準備
# 1 データ準備
# 2 プロット確認
# 3 モデル構築
# 4 モデル訓練
# 5 モデル精度の評価
# 6 予測データの作成
# 7 プロット出力



# 0 環境準備 -------------------------------------------------------------------------------

# ライブラリ
library(reticulate)
library(magrittr)
library(tidyverse)
library(keras)


# 仮想環境の選択
use_condaenv("C:/Users/Owner/Anaconda3/envs/r-reticulate", required = TRUE)
py_config()



# 1 データ準備 -------------------------------------------------------------------------------

# データロード
# --- Fashion MNISTデータ
fashion_mnist <- dataset_fashion_mnist()


# データ構造
fashion_mnist %>% glimpse()


# データ準備
c(train_images, train_labels) %<-% fashion_mnist$train
c(test_images, test_labels) %<-% fashion_mnist$test


# 分類ラベルの定義
class_names = c('T-shirt/top',
                'Trouser',
                'Pullover',
                'Dress',
                'Coat',
                'Sandal',
                'Shirt',
                'Sneaker',
                'Bag',
                'Ankle boot')


# データ確認
# --- 訓練データ
train_images %>% dim()
train_labels %>% dim()
train_labels[1:20]


# データ確認
# --- テストデータ
test_images %>% dim()
test_labels %>% dim()
test_labels[1:20]


# データ変換
# --- 3つのピクセルの値は0〜255の整数であり、0〜1の浮動小数点数に変換する
train_images <- train_images / 255
test_images <- test_images / 255




# 2 プロット確認 -------------------------------------------------------------------------------

# イメージデータの抽出
# --- プロット用に加工
image_1 <-
  train_images[1, , ] %>%
    as.data.frame() %>%
    set_colnames(seq_len(ncol(.))) %>%
    mutate(y = seq_len(nrow(.))) %>%
    gather( "x", "value", -y) %>%
    mutate(x = as.integer(x))


# データ確認
image_1 %>% glimpse()


# プロット
# --- 個別イメージ
image_1 %>%
  ggplot(aes(x = x, y = y, fill = value)) +
    geom_tile() +
    scale_fill_gradient(low = "white", high = "black", na.value = NA) +
    scale_y_reverse() +
    theme_minimal() +
    theme(panel.grid = element_blank())   +
    theme(aspect.ratio = 1) +
    xlab("") +
    ylab("")


# プロット
# --- 全イメージ
par(mfcol = c(5, 5))
par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')
for (i in 1:25) {
  img <- train_images[i, , ]
  img <- t(apply(img, 2, rev))
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste(class_names[train_labels[i] + 1]))
}



# 3 モデル構築 -------------------------------------------------------------------------------

# ＜ポイント＞
# - ニューラルネットワークの基本的な構成要素はレイヤーで、レイヤーに入力されたデータから表現を抽出する
# - ディープラーニングのほとんどは、単純なレイヤーをチェーン化することで構成される


# モデル準備
model <- keras_model_sequential()


# ネットワーク構築
# --- 最初のレイヤーinput_shapeは入力の次元を表す引数を指定する必要がある
# --- 今回の場合、画像は28x28です。
model %>%
  layer_flatten(input_shape = c(28, 28)) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')


# コンパイル
# --- ｢損失関数｣｢オプティマイザー｣｢メトリック｣の設定
model %>% compile(
  optimizer = 'adam',
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)


# 確認
model %>% print()
model %>% class()



# 4 モデル訓練 -------------------------------------------------------------------------------

# モデルトレーニング
# --- 訓練データ
model %>% fit(x = train_images, y = train_labels, epochs = 5, verbose = 2)



# 5 モデル精度の評価 -------------------------------------------------------------------------------

# モデル精度の評価
score <- model %>% evaluate(test_images, test_labels, verbose = 0)


# 確認
score %>% print()
score %>% glimpse()



# 6 予測データの作成 -------------------------------------------------------------------------------

# 予測データの作成
# --- クラス確率が出力される
predictions <- model %>% predict(test_images)
predictions %>% glimpse()


# 分類結果の取得
# --- クラス確率が最大となる位置を取得
predictions[1, ]
which.max(predictions[1, ])


# 分類結果の取得
# --- 直接取得
class_pred <- model %>% predict_classes(test_images)
class_pred[1:20]



# 7 プロット出力 -------------------------------------------------------------------------------

# 予測結果をプロット出力
par(mfcol=c(5,5))
par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')
for (i in 1:25) {
  img <- test_images[i, , ]
  img <- t(apply(img, 2, rev))
  # subtract 1 as labels go from 0 to 9
  predicted_label <- which.max(predictions[i, ]) - 1
  true_label <- test_labels[i]
  if (predicted_label == true_label) {
    color <- '#008800'
  } else {
    color <- '#bb0000'
  }
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste0(class_names[predicted_label + 1], " (",
                      class_names[true_label + 1], ")"),
        col.main = color)
}




