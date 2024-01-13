library(reticulate)
set.seed(1)


sim_SBM_list <- function(num_seq=10, n=50, rho=0){
  
  output <- list()
  for(rep_iter in 1:num_seq){
    
    K = 3
    v =  c(26, 51, 76)
    num_time = 100
    df = vector(mode = "list", length = num_time)
    
    for(t in 1:num_time){
      
      if( t==1 || t==v[2] ){ # t=1 or t=51
        
        P =  matrix(0.3,n,n)
        P[1:floor(n/K), 1:floor(n/K)] = 0.5
        P[(1+floor(n/K)):(2*floor(n/K)),(1+floor(n/K)):(2*floor(n/K)) ] = 0.5
        P[(1+2*floor(n/K)):n,(1+2*floor(n/K)):n ] = 0.5
        diag(P) = 0
        
        A = matrix(rbinom(matrix(1,n,n),matrix(1,n,n),P),n,n)
        
      }
      
      if(t == v[1] || t == v[3]){ # t=26 or t=76
        
        Q =  matrix(0.2,n,n)
        Q[1:floor(n/K), 1:floor(n/K)] = 0.45
        Q[(1+floor(n/K)):(2*floor(n/K)),(1+floor(n/K)):(2*floor(n/K)) ] = 0.45
        Q[(1+2*floor(n/K)):n,(1+2*floor(n/K)):n ] = 0.45
        diag(Q) = 0
        
        A = matrix(rbinom(matrix(1,n,n),matrix(1,n,n),Q),n,n)
        
      }
      
      if( (t > 1 && t < v[1])  ||  (t > v[2] && t < v[3]) ){ # t=2 to t=25 or t=52 to t=75
        
        aux1 = (1-P)*rho + P # (1-E(t+1))*rho + E(t+1) if A(t) = 1
        aux2 = P*(1-rho) # (1-E(t+1))*rho + E(t+1) if A(t) = 0
        
        aux1 = matrix(rbinom(matrix(1,n,n),matrix(1,n,n),aux1),n,n)
        aux2 = matrix(rbinom(matrix(1,n,n),matrix(1,n,n),aux2),n,n)
        A =  aux1*A + aux2*(1-A)
        
      }
      
      if( (t > v[1] && t < v[2]) || ((t > v[3] && t <= num_time)) ){ # t=27 to t=50 or t=77 to t=100
        
        aux1 = (1-Q)*rho + Q
        aux2 = Q*(1-rho)
        
        aux1 = matrix(rbinom(matrix(1,n,n),matrix(1,n,n),aux1),n,n)
        aux2 = matrix(rbinom(matrix(1,n,n),matrix(1,n,n),aux2),n,n)
        A =  aux1*A + aux2*(1-A)
        
      }
      
      diag(A) <- 0
      df[[t]] = A
      
    }
    
    output[[rep_iter]] <- df
    
  }
  
  return(output)
}


save_to_numpy <- function(data_list, file_name){
  num_seq <- length(data_list)
  num_time <- length(data_list[[1]])
  num_node <- dim(data_list[[1]][[1]])[1] # first dim of matrix
  
  output <- array(NA, dim = c(num_seq, num_time, num_node, num_node))
  
  for(seq_iter in 1:num_seq){
    for(time_iter in 1:num_time){
      output[seq_iter, time_iter,,] <- data_list[[seq_iter]][[time_iter]]
    }
  }
  np = import("numpy")
  np$save(file_name, r_to_py(output))
  print('File saved')
}





#################
# rho=0.5, n=50 #
#################
SBM_seq10T100n50rho05 <- sim_SBM_list(num_seq=10, n=50, rho=0.5)
save(SBM_seq10T100n50rho05, file = "SBM_seq10T100n50rho05.RData") # R
save_to_numpy(SBM_seq10T100n50rho05, "SBM_seq10T100n50rho05.npy") # Python



##################
# rho=0.5, n=100 #
##################
SBM_seq10T100n100rho05 <- sim_SBM_list(num_seq=10, n=100, rho=0.5)
save(SBM_seq10T100n100rho05, file = "SBM_seq10T100n100rho05.RData") # R
save_to_numpy(SBM_seq10T100n100rho05, "SBM_seq10T100n100rho05.npy") # Python
