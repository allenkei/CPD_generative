library(tergm)
library(dplyr)
library(reticulate)
set.seed(1)

sim_STERGM_list <- function(num_seq=10, n=50, network_stats,
                            coefs_pos, coefs_neg, y1_stats, node_attr=NA){
  num_nodes <- n
  num_timepts <- 100
  num_changepts <- 3
  form_model <- diss_model <- as.formula(paste("~", paste(network_stats, collapse = "+")))
  
  output <- list()
  for(rep_iter in 1:num_seq){
    
    # generate initial network
    g0<-network.initialize(num_nodes, directed=T) # empty
    g1<-san(g0~edges,target.stats=y1_stats, verbose=TRUE)
    if(any(!is.na(node_attr))){
      network::set.vertex.attribute(g1, "Gender", node_attr)
    }
    ginit <- g1
    
    time_stable <- num_timepts / (num_changepts+1)
    cur_end <- 0
    res_adj_list <- vector(mode = 'list', length = num_timepts)
    for(i in 1:(num_changepts+1)){
      cat('[INFO] Simulate from ', cur_end+1, ' to ', cur_end + time_stable, '\n')
      stergm.sim <- simulate(
        ginit,
        formation=form_model,
        dissolution=diss_model,
        coef.form=coefs_pos[,i],
        coef.diss=coefs_neg[,i],
        nsim = 1,
        time.slices = time_stable,
        time.start = cur_end # needs to be update every time a new dynamic is generated
      )
      # newstart_nw%n%'net.obs.period' check https://github.com/statnet/tergm/blob/master/R/simulate.stergm.R for details
      
      for(t in (1 + cur_end) : (time_stable + cur_end)){
        tmpnet <- network.extract(stergm.sim, at = t) %>% network()
        res_adj_list[[t]] <- tmpnet[, ]
      }
      cur_end <- cur_end + time_stable
      ginit <- network.extract(stergm.sim, at = cur_end) %>% network()
      
    }
    
    output[[rep_iter]] <- res_adj_list
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



#######
# p=6 #
#######
network_stats=c("edges", "mutual", "triangles")
coefs_pos <- matrix(c(-2, -1.5, -2, -1.5,
                       2,  1,    2,  1,
                      -2, -1,   -2, -1),
                    nrow=3, ncol=4, byrow = T)

coefs_neg <- matrix(c( -1,  2,    -1,  2,
                        2,  1,     2,  1,
                        1,  1.5,   1,  1.5),
                    nrow=3, ncol=4, byrow = T)

STERGM_seq10T100n50p6 <- sim_STERGM_list(num_seq=10, n=50, network_stats, 
                               coefs_pos, coefs_neg, y1_stats=500)

save(STERGM_seq10T100n50p6, file = "STERGM_seq10T100n50p6.RData") # R
save_to_numpy(STERGM_seq10T100n50p6, "STERGM_seq10T100n50p6.npy") # Python


STERGM_seq10T100n100p6 <- sim_STERGM_list(num_seq=10, n=100, network_stats, 
                                         coefs_pos, coefs_neg, y1_stats=1400)

save(STERGM_seq10T100n100p6, file = "STERGM_seq10T100n100p6.RData") # R
save_to_numpy(STERGM_seq10T100n100p6, "STERGM_seq10T100n100p6.npy") # Python

