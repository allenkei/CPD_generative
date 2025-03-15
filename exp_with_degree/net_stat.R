library(reticulate)
library(ergm)
library(reshape2)
library(ggplot2)
library(patchwork)


#######################
# DEGREE DISTRIBUTION #
#######################

np <- import("numpy") 
adj_prob <- np$load(file.choose()) # adj_prob_seq0.npy
original_data <- np$load(file.choose()) # data

dim(adj_prob) # 18000, 100, 100
dim(original_data) # 10, 100, 100, 100
 
n_seq <- 1
max_bin <- 80 # STERGM=30, SBM=60, NN=80

# Also, T=90, m=200



process_graph <- function(prob_matrix) {
  n <- dim(prob_matrix)[1]
  diag(prob_matrix) <- 0
  bernoulli_matrix <- matrix(rbinom(n * n, 1, prob_matrix), nrow = n)
  return(bernoulli_matrix)
}

stat_holder <- matrix(0, nrow=dim(adj_prob)[1], ncol=max_bin)
true_stat_holder <- matrix(0, nrow=100, ncol=max_bin)


for (i in 1:dim(adj_prob)[1]){
  graph_i <- process_graph(adj_prob[i,,]) # randomly sampled
  degrees <- rowSums(graph_i)
  stat_holder[i,] <- tabulate(degrees + 1, nbins = max_bin)
};rm(graph_i,i,degrees)



for (i in 1:100){
  graph_i <- original_data[n_seq,i,,]
  degrees <- rowSums(graph_i)
  true_stat_holder[i,] <- tabulate(degrees + 1, nbins = max_bin)
};rm(graph_i,i,degrees)





last_t <- seq(9, 90, by = 9)
true_t <- seq(10, 100, by = 10)
plots <- list()


for(i in 1:10){
  
  rows_selection <- ((last_t[i]-1) * 200 + 1):( last_t[i]* 200)  # using mu at t=9 to predict y at t=10
  t_selection <- true_t[i]                   
  
  degree_df <- as.data.frame(stat_holder[rows_selection, ])
  colnames(degree_df) <- paste0(0:(max_bin-1))
  degree_long <- melt(degree_df, variable.name = "Degree", value.name = "Count")
  
  true_graph_df <- data.frame(
    Degree = factor(paste0(0:(max_bin-1)), levels = paste0(0:(max_bin-1))),
    Count = true_stat_holder[t_selection, ]
  )
  
  plot <- ggplot(degree_long, aes(x = Degree, y = Count)) +
    geom_boxplot() +
    labs(title = paste("t =", t_selection), x = "Degree", y = "Count") +
    geom_line(data = true_graph_df, aes(x = Degree, y = Count, group = 1), 
              color = "red", linewidth = 1) + 
    scale_x_discrete(breaks = paste0(seq(0, max_bin, by = 10)), limits = paste0(0:max_bin)) + 
    scale_y_continuous(breaks = seq(0, 20, by = 5), limits = c(0, 20))
  
    plots[[length(plots) + 1]] <- plot
}


grid_plot <- wrap_plots(plots, nrow = 2, ncol = 5)

print(grid_plot)





#############################
# EDGE-WISE SHARED PARTNERS #
#############################

np <- import("numpy") 
adj_prob <- np$load(file.choose()) # adj_prob_seq0.npy
original_data <- np$load(file.choose()) # data

dim(adj_prob) # 18000, 100, 100
dim(original_data) # 10, 100, 100, 100

n_seq <- 1
max_bin <- 10 # STERGM=10, SBM=26, NN=50

# Also, T=90, m=200


process_graph <- function(prob_matrix) {
  n <- dim(prob_matrix)[1]
  diag(prob_matrix) <- 0
  bernoulli_matrix <- matrix(rbinom(n * n, 1, prob_matrix), nrow = n)
  return(bernoulli_matrix)
}

stat_holder <- matrix(0, nrow=dim(adj_prob)[1], ncol=max_bin)
true_stat_holder <- matrix(0, nrow=100, ncol=max_bin)



for (i in 1:dim(adj_prob)[1]){
  graph_i <- process_graph(adj_prob[i,,]) # randomly sampled
  stat_holder[i,] <- summary(graph_i ~ esp(1:max_bin))
};rm(graph_i,i)



for (i in 1:100){
  graph_i <- original_data[n_seq,i,,]
  true_stat_holder[i,] <- summary(graph_i ~ esp(1:max_bin))
};rm(graph_i,i)





last_t <- seq(9, 90, by = 9)
true_t <- seq(10, 100, by = 10)
plots <- list()


for(i in 1:10){
  
  rows_selection <- ((last_t[i]-1) * 200 + 1):( last_t[i]* 200)  # using mu at t=9 to predict y at t=10
  t_selection <- true_t[i]                   
  
  esp_df <- as.data.frame(stat_holder[rows_selection, ])
  colnames(esp_df) <- paste0(1:max_bin)
  esp_long <- melt(esp_df, variable.name = "ESP", value.name = "Count")
  
  true_graph_df <- data.frame(
    ESP = factor(paste0(1:max_bin), levels = paste0(1:max_bin)),
    Count = true_stat_holder[t_selection, ]
  )
  
  plot <- ggplot(esp_long, aes(x = ESP, y = Count)) +
    geom_boxplot() +
    labs(title = paste("t =", t_selection), x = "Edge-wise Shared Partners", y = "Count") +
    geom_line(data = true_graph_df, aes(x = ESP, y = Count, group = 1), 
              color = "red", linewidth = 1) +
    scale_x_discrete(breaks = paste0(seq(0, max_bin, by = 2)), limits = paste0(0:max_bin)) # interval: 2, 4, 10
  
  plots[[length(plots) + 1]] <- plot
}


grid_plot <- wrap_plots(plots, nrow = 2, ncol = 5)

# 12 by 6
print(grid_plot)


