library(nett)
library(Matrix)
library(CPDstergm)
data("MITphone")

set.seed(1)
n <- dim(MITphone[[1]])[1]
num_T <- length(MITphone)


est_CP <- c(39,  94, 126, 161, 200)
nbs_result <- c(46,  94, 126, 158, 178, 198)
rdpg_result <- c(94, 114, 178)
result_est_CP <- c(29,  40,  49,  63, 94, 191)
kerSeg_result <- c(34,  94, 113, 200)
gSeg_result <- c(48,  88,  95, 118, 178)




#A_seq <- MITphone
#change_points <- est_CP


#change_points <- c(change_points, length(A_seq) + 1)
#log_likelihood <- 0


#start <- 1
#cp_idx <- 1





cal_log_likelihood <- function(A_seq, change_points, excluded_indices) {
  
  change_points <- c(change_points,num_T+1)
  log_likelihood <- 0
  
  start <- 1 # cp_idx <- 5
  for (cp_idx in seq_along(change_points)) {
    end <- change_points[cp_idx] - 1
    
    segment_indices <- setdiff(start:end, excluded_indices)
    excluded_in_segment <- intersect(start:end, excluded_indices)
    
    if (length(segment_indices) > 0) {
      
      A_bar <- Reduce("+", A_seq[segment_indices]) / length(segment_indices)  # Mean adjacency matrix
      A_bar <- ifelse(A_bar > 0.1, 1, 0)
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




gap_choice <- c(15,20,25,30)


log_lik_holder <- matrix(NA, nrow=6, ncol = length(gap_choice))


for(iter in 1:length(gap_choice)){
  
  gap <- gap_choice[iter]
  excluded_indices=seq(gap,230,by=gap)
  
  log_lik_holder[1,iter] <- cal_log_likelihood(MITphone, est_CP, excluded_indices)
  log_lik_holder[2,iter] <- cal_log_likelihood(MITphone, nbs_result, excluded_indices)
  log_lik_holder[3,iter] <- cal_log_likelihood(MITphone, rdpg_result, excluded_indices)
  log_lik_holder[4,iter] <- cal_log_likelihood(MITphone, result_est_CP, excluded_indices)
  log_lik_holder[5,iter] <- cal_log_likelihood(MITphone, kerSeg_result, excluded_indices)
  log_lik_holder[6,iter] <- cal_log_likelihood(MITphone, gSeg_result, excluded_indices)
  
}




t(log_lik_holder)


c(which.max(log_lik_holder[,1]), which.max(log_lik_holder[,2]), which.max(log_lik_holder[,3]), which.max(log_lik_holder[,4]))





