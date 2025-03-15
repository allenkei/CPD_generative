library(nett)
library(Matrix)
library(reticulate)
np <- import("numpy")


data <- np$load(file.choose()) # Enron.npy

Enron <- list()
for(iter in 1:dim(data)[1]){ Enron[[iter]] <- data[iter,,] }


set.seed(1)
n <- dim(Enron[[1]])[1]
num_T <- length(Enron)


est_CP <- c(20, 54, 69, 79)
nbs_result <- c(14, 36, 54, 66, 78)
rdpg_result <- c(19, 32, 51, 63, 68, 80, 88)
result_est_CP <- c(64, 72, 80, 88)
kerSeg_result <- c(6, 21, 51, 89, 95)
gSeg_result <- c(6, 20, 83, 89, 95)



#A_seq <- Enron
#change_points <- est_CP


#change_points <- c(change_points, length(A_seq) + 1)
#log_likelihood <- 0


#start <- 1
#cp_idx <- 1





cal_log_likelihood <- function(A_seq, change_points, excluded_indices) {
  
  change_points <- c(change_points,num_T+1)
  log_likelihood <- 0
  
  start <- 1
  for (cp_idx in seq_along(change_points)) {
    end <- change_points[cp_idx] - 1
    
    segment_indices <- setdiff(start:end, excluded_indices)
    excluded_in_segment <- intersect(start:end, excluded_indices)
    
    if (length(segment_indices) > 0) {
      
      A_bar <- Reduce("+", A_seq[segment_indices]) / length(segment_indices)  # Mean adjacency matrix
      A_bar <- ifelse(A_bar > 0.05, 1, 0)
      A_bar <- A_bar + diag(n)
      
      
      K_choices <- c(2,3,4,5)
      BIC_holder <- numeric(length(K_choices))
      for(k_iter in 1:length(K_choices)){
        K <- K_choices[k_iter]
        label <- fast_cpl(Matrix(A_bar), K)
        BIC_holder[k_iter] <- eval_dcsbm_bic(Matrix(A_bar), label, K, poi=T)
      }
      
      # after selection
      K <- K_choices[which.min(BIC_holder)]
      label <- fast_cpl(Matrix(A_bar), K)
      
      
      for (excluded_idx in excluded_in_segment) {
        adj <- A_seq[[excluded_idx]] + diag(n)
        log_likelihood <- log_likelihood + eval_dcsbm_like(Matrix(adj), label, poi = TRUE)
      }
    }
    
    start <- end + 1
  }
  
  return(log_likelihood)
}




gap_choice <- c(6,8,10,12)


log_lik_holder <- matrix(NA, nrow=6, ncol = length(gap_choice))


for(iter in 1:length(gap_choice)){
  
  gap <- gap_choice[iter]
  excluded_indices=seq(gap,100,by=gap)
  
  log_lik_holder[1,iter] <- cal_log_likelihood(Enron, est_CP, excluded_indices)
  log_lik_holder[2,iter] <- cal_log_likelihood(Enron, nbs_result, excluded_indices)
  log_lik_holder[3,iter] <- cal_log_likelihood(Enron, rdpg_result, excluded_indices)
  log_lik_holder[4,iter] <- cal_log_likelihood(Enron, result_est_CP, excluded_indices)
  log_lik_holder[5,iter] <- cal_log_likelihood(Enron, kerSeg_result, excluded_indices)
  log_lik_holder[6,iter] <- cal_log_likelihood(Enron, gSeg_result, excluded_indices)
  
}


t(log_lik_holder)


c(which.max(log_lik_holder[,1]), which.max(log_lik_holder[,2]), which.max(log_lik_holder[,3]), which.max(log_lik_holder[,4]))






