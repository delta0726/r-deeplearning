# ***************************************************************************************
# Title     : Tensors and operations
# Objective : TODO
# Created by: Owner
# Created on: 2020/11/29
# URL       : https://tensorflow.rstudio.com/tutorials/advanced/customization/tensors-operations/
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
# 1 テンソル演算
# 2 Rの配列の互換性
# 3 GPUアクセラレーション



# 0 環境準備 -------------------------------------------------------------------------------

# ライブラリ
library(reticulate)
library(tensorflow)
library(keras)
library(tidyverse)


# 仮想環境の選択
use_condaenv("C:/Users/Owner/Anaconda3/envs/r-reticulate", required = TRUE)
py_config()



# 1. テンソル演算 -------------------------------------------------------------------------------

# ＜ポイント＞
# - テンソルは多次元配列
# - Rのarrayオブジェクトと同様に、tf$Tensorオブジェクトにはデータ型(dtype)と形状(shape)があります
# - tf$TensorはGPUのようなアクセラレータメモリに常駐できる


# 基本演算
tf$add(1, 2)
tf$add(c(1, 2), c(3, 4))
tf$square(5)
tf$reduce_sum(c(1, 2, 3))


# 基本演算子
tf$square(2) + tf$square(3)


# 行列
x = tf$matmul(matrix(1,ncol = 1), matrix(c(2, 3), nrow = 1))
x %>% print()
x$shape
x$dtype



# 2. Rの配列の互換性 -------------------------------------------------------------------------------

# ＜ポイント＞
# - Rの配列とtf.Tensorsの間には互換性があり変換が容易
# - TensorFlowは、R配列をTensorに自動的に変換
# - テンソルは、as.array、as.matrix、またはas.numericメソッドを使用して明示的にR配列に変換


# TensorFlow operations convert arrays to Tensors automatically
1 + tf$ones(shape = 1)

# The as.array method explicitly converts a Tensor to an array
as.array(tf$ones(shape = 1))




# 3. GPUアクセラレーション -------------------------------------------------------------------------------

x <- tf$random$uniform(shape(3, 3))

# List devices
tf$config$experimental$list_physical_devices()

# What device is x placed
x$device




print("On CPU:0:")
with(tf$device("CPU:0"), {
  x <- tf$ones(shape(1000, 1000))
  print(x$device)
})

print("On GPU:0:")
with(tf$device("GPU:0"), {
  x <- tf$ones(shape(1000, 1000))
  print(x$device)
})