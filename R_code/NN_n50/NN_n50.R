library(CPDstergm)
source("EVAL.R") # FILE DIRECTORY

library(reticulate)
np <- import("numpy")
np_data <- np$load("NN_seq10T100n50.npy") # DATA DIRECTORY
dim(np_data) 

NN_seq10T100n50 <- list()

for(seq_iter in 1:10){
  NN_one_seq <- np_data[seq_iter,,,]
  temp = list()
  for(time_t in 1:100){
    temp[[time_t]] <- NN_one_seq[time_t,,]
  }
  
  NN_seq10T100n50[[seq_iter]] <- temp
};rm(temp, NN_one_seq, np_data)


result_n50 <- matrix(0, nrow=7, ncol=4)


network_stats=c("edges", "mutual")
sim_result1 <- CPD_STERGM_list(NN_seq10T100n50, directed=TRUE, network_stats)
result_n50[1,] <- colMeans(sim_result1)


network_stats=c("edges", "mutual", "triangles")
sim_result2 <- CPD_STERGM_list(NN_seq10T100n50, directed=TRUE, network_stats)
result_n50[2,] <- colMeans(sim_result2)


sim_result3 <- Evaluation_gSeg(NN_seq10T100n50, p_threshold=0.05)
result_n50[3,] <- colMeans(sim_result3)


sim_result4 <- Evaluation_kerSeg(NN_seq10T100n50, p_threshold=0.001)
result_n50[4,] <- colMeans(sim_result4)


sim_result5 <- Evaluation_gSeg_on_stats(NN_seq10T100n50, p_threshold=0.05, num_stats=3)
result_n50[5,] <- colMeans(sim_result5)


sim_result6 <- Evaluation_kerSeg_on_stats(NN_seq10T100n50, p_threshold=0.001, num_stats=3)
result_n50[6,] <- colMeans(sim_result6)


sim_result7 <- Evaluation_RDPG(NN_seq10T100n50, M=50, d=5, delta=5)
result_n50[7,] <- colMeans(sim_result7)

write.csv(result_n50, 'result_n50.csv')



# print result
apply(sim_result1, 2, mean)
apply(sim_result1, 2, sd) * sqrt(9/10)

apply(sim_result2, 2, mean)
apply(sim_result2, 2, sd) * sqrt(9/10)

apply(sim_result3, 2, mean)
apply(sim_result3, 2, sd) * sqrt(9/10)

apply(sim_result4, 2, mean)
apply(sim_result4, 2, sd) * sqrt(9/10)

apply(sim_result5, 2, mean)
apply(sim_result5, 2, sd) * sqrt(9/10)

apply(sim_result6, 2, mean)
apply(sim_result6, 2, sd) * sqrt(9/10)

apply(sim_result7, 2, mean)
apply(sim_result7, 2, sd) * sqrt(9/10)


