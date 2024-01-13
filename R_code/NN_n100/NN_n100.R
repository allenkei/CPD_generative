library(CPDstergm)
source("EVAL.R") # FILE DIRECTORY



library(reticulate)
np <- import("numpy")
np_data <- np$load("/NN_seq10T100n100.npy") # FILE DIRECTORY
dim(np_data) 

NN_seq10T100n100 <- list()

for(seq_iter in 1:10){
  NN_one_seq <- np_data[seq_iter,,,]
  temp = list()
  for(time_t in 1:100){
    temp[[time_t]] <- NN_one_seq[time_t,,]
  }
  
  NN_seq10T100n100[[seq_iter]] <- temp
}


result_n100 <- matrix(0, nrow=6, ncol=4)


network_stats=c("edges", "mutual")
sim_result1 <- CPD_STERGM_list(NN_seq10T100n100, directed=TRUE, network_stats)
result_n100[1,] <- colMeans(sim_result1)


network_stats=c("edges", "mutual", "triangles")
sim_result2 <- CPD_STERGM_list(NN_seq10T100n100, directed=TRUE, network_stats)
result_n100[2,] <- colMeans(sim_result2)


sim_result3 <- Evaluation_gSeg(NN_seq10T100n100, p_threshold=0.05)
result_n100[3,] <- colMeans(sim_result3)


sim_result4 <- Evaluation_kerSeg(NN_seq10T100n100, p_threshold=0.001)
result_n100[4,] <- colMeans(sim_result4)


sim_result5 <- Evaluation_gSeg_on_stats(NN_seq10T100n100, p_threshold=0.05, num_stats=3)
result_n100[5,] <- colMeans(sim_result5)


sim_result6 <- Evaluation_kerSeg_on_stats(NN_seq10T100n100, p_threshold=0.001, num_stats=3)
result_n100[6,] <- colMeans(sim_result6)


write.csv(result_n100, 'result_n100.csv')


# visualization (9 by 6 inches)

par(mfrow=c(2,3))

for(i in c(25,50,75, 26,51,76)){
  par(mar = c(2.2, 2.2, 2.2, 2.2)); 
  image(NN_seq10T100n100[[1]][[i]], xaxt = "n", yaxt = "n")
  axis(side=1,at=seq(0,1,length.out = 4),labels=round(seq(1,100,length.out = 4)),xpd=NA,cex.axis=1.4)
  axis(side=2,at=seq(0,1,length.out = 4),labels=round(seq(1,100,length.out = 4)),xpd=NA,cex.axis=1.4)
}

