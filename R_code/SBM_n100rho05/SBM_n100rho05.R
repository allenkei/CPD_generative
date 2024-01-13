
library(devtools)
install_github("allenkei/CPDstergm")
library(CPDstergm)
source("EVAL.R") # FILE DIRECTORY

load("SBM_seq10T100n100rho05.RData") # DATA DIRECTORY
result_n100rho05 <- matrix(0, nrow=6, ncol=4)


network_stats=c("edges", "mutual")
sim_result1 <- CPD_STERGM_list(SBM_seq10T100n100rho05, directed=TRUE, network_stats)
result_n100rho05[1,] <- colMeans(sim_result1) # apply(sim_result1, 2, median)


network_stats=c("edges", "mutual", "triangles")
sim_result2 <- CPD_STERGM_list(SBM_seq10T100n100rho05, directed=TRUE, network_stats)
result_n100rho05[2,] <- colMeans(sim_result2) # apply(sim_result2, 2, median)


sim_result3 <- Evaluation_gSeg(SBM_seq10T100n100rho05, p_threshold=0.05)
result_n100rho05[3,] <- colMeans(sim_result3)


sim_result4 <- Evaluation_kerSeg(SBM_seq10T100n100rho05, p_threshold=0.001)
result_n100rho05[4,] <- colMeans(sim_result4)


sim_result5 <- Evaluation_gSeg_on_stats(SBM_seq10T100n100rho05, p_threshold=0.05, num_stats=3)
result_n100rho05[5,] <- colMeans(sim_result5)


sim_result6 <- Evaluation_kerSeg_on_stats(SBM_seq10T100n100rho05, p_threshold=0.001, num_stats=3)
result_n100rho05[6,] <- colMeans(sim_result6)


write.csv(result_n100rho05, 'result_n100rho05.csv')




# visualization (9 by 6 inches)

par(mfrow=c(2,3))

for(i in c(25,50,75, 26,51,76)){
  par(mar = c(2.2, 2.2, 2.2, 2.2)); 
  image(SBM_seq10T100n100rho05[[1]][[i]], xaxt = "n", yaxt = "n")
  axis(side=1,at=seq(0,1,length.out = 4),labels=round(seq(1,100,length.out = 4)),xpd=NA,cex.axis=1.4)
  axis(side=2,at=seq(0,1,length.out = 4),labels=round(seq(1,100,length.out = 4)),xpd=NA,cex.axis=1.4)
}


