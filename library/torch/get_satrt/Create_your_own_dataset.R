# ***************************************************************************************
# Library   : torch
# Title     : Create your own Dataset
# Created by: Owner
# Created on: 2021/08/21
# URL       : https://torch.mlverse.org/start/custom_dataset/
# ***************************************************************************************


# ＜概要＞
# - {torch}ではデータセットをミニバッチで処理するためジェネレータを定義する必要がある
# - torchはすべてのデータが数値形式である必要ある（文字型は因子変換）


# ＜参考＞
# Pytorch – DataLoader の使い方について解説
# - https://pystyle.info/pytorch-dataloader/


# ＜目次＞
# 0 準備
# 1 文字列のテンソル変換
# 2 データセットの変換定義
# 3 データ変換
# 4 データローダーの定義
# 5 モデル構築
# 6 学習


# 0 準備 ------------------------------------------------------------------------

# ライブラリ
library(dplyr)
library(torch)
library(palmerpenguins)



# データ確認
# --- ペンギンの生体情報
# --- 特徴量は数値とファクターを含む（ラベルはspecies）
penguins %>% print()
penguins %>% glimpse()



# 1 文字列のテンソル変換 --------------------------------------------------------

# 数値型
torch_tensor(1)

# 文字列
"one" %>% 
  as.factor() %>% 
  as.numeric() %>% 
  as.integer() %>% 
  torch_tensor()


# 2 データセットの変換定義 -------------------------------------------------------

penguins_dataset <- dataset(
  
  name = "penguins_dataset",
  
  initialize = function(df) {
    
    df <- na.omit(df) 
    
    # continuous input data (x_cont)   
    x_cont <- df[ , c("bill_length_mm", "bill_depth_mm", 
                      "flipper_length_mm", "body_mass_g", "year")] %>% as.matrix()

    self$x_cont <- torch_tensor(x_cont)
    
    # categorical input data (x_cat)
    x_cat <- df[ , c("island", "sex")]
    x_cat$island <- as.integer(x_cat$island)
    x_cat$sex <- as.integer(x_cat$sex)
    self$x_cat <- as.matrix(x_cat) %>% torch_tensor()
    
    # target data (y)
    species <- as.integer(df$species)
    self$y <- torch_tensor(species)
    
  },
  
  .getitem = function(i) {
    list(x_cont = self$x_cont[i, ], x_cat = self$x_cat[i, ], y = self$y[i])
    
  },
  
  .length = function() {
    self$y$size()[[1]]
  }
  
)


# 3 データ変換 ---------------------------------------------------------------

# 行番号の抽出
# --- 250レコードをランダムに抽出
train_indices <- sample(1:nrow(penguins), 250)
valid_indices <- setdiff(1:nrow(penguins), train_indices)

# データ抽出
train_ds <- penguins %>% slice(train_indices) %>% penguins_dataset()
valid_ds <- penguins %>% slice(valid_indices) %>% penguins_dataset()

# レコード数
train_ds %>% length()
valid_ds %>% length()

# データ確認
train_ds %>% print()
train_ds[1] %>% print()


# 4 データローダーの定義-------------------------------------------------------

# データローダーの作成
# --- Dataset からサンプルを取得してミニバッチを作成するクラス
train_dl <- train_ds %>% dataloader(batch_size = 16, shuffle = TRUE)
valid_dl <- valid_ds %>% dataloader(batch_size = 16, shuffle = FALSE)


# 確認
train_dl %>% print()
valid_dl %>% print()


# 5 モデル構築 -----------------------------------------------------------------

# 畳み込みの定義
embedding_module <- nn_module(
  
  initialize = function(cardinalities) {
    
    self$embeddings = nn_module_list(lapply(cardinalities, function(x) nn_embedding(num_embeddings = x, embedding_dim = ceiling(x/2))))
    
  },
  
  forward = function(x) {
    
    embedded <- vector(mode = "list", length = length(self$embeddings))
    for (i in 1:length(self$embeddings)) {
      embedded[[i]] <- self$embeddings[[i]](x[ , i])
    }
    
    torch_cat(embedded, dim = 2)
  }
)


# ネットワーク構築
net <- nn_module(
  "penguin_net",
  
  initialize = function(cardinalities,
                        n_cont,
                        fc_dim,
                        output_dim) {
    
    self$embedder <- embedding_module(cardinalities)
    self$fc1 <- nn_linear(sum(purrr::map(cardinalities, function(x) ceiling(x/2)) %>% unlist()) + n_cont, fc_dim)
    self$output <- nn_linear(fc_dim, output_dim)
    
  },
  
  forward = function(x_cont, x_cat) {
    
    embedded <- self$embedder(x_cat)
    
    all <- torch_cat(list(embedded, x_cont$to(dtype = torch_float())), dim = 2)
    
    all %>% self$fc1() %>%
      nnf_relu() %>%
      self$output() %>%
      nnf_log_softmax(dim = 2)
    
  }
)


# モデル構築
model <- net(
  cardinalities = c(length(levels(penguins$island)), length(levels(penguins$sex))),
  n_cont = 5,
  fc_dim = 32,
  output_dim = 3
)


# オプティマイザー設定
optimizer <- optim_adam(model$parameters, lr = 0.01)



# 6 学習 ----------------------------------------------------------------

epoch <- 1
for (epoch in 1:20) {
  
  model$train()
  train_losses <- c()  
  
  coro::loop(for (b in train_dl) {
    
    optimizer$zero_grad()
    output <- model(b$x_cont, b$x_cat)
    loss <- nnf_nll_loss(output, b$y)
    
    loss$backward()
    optimizer$step()
    
    train_losses <- c(train_losses, loss$item())
    
  })
  
  model$eval()
  valid_losses <- c()
  
  coro::loop(for (b in valid_dl) {
    
    output <- model(b$x_cont, b$x_cat)
    loss <- nnf_nll_loss(output, b$y)
    valid_losses <- c(valid_losses, loss$item())
    
  })
  
  cat(sprintf("Loss at epoch %d: training: %3.3f, validation: %3.3f\n", epoch, mean(train_losses), mean(valid_losses)))
}
