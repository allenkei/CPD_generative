
library(devtools)
install_github("allenkei/CPDstergm")
library(CPDstergm)
source("EVAL.R") # FILE DIRECTORY

load("~/CPD_nn/SBM_seq10T100n50rho05.RData") # DATA DIRECTORY
result_n50rho05 <- matrix(0, nrow=6, ncol=4)


network_stats=c("edges", "mutual")
sim_result1 <- CPD_STERGM_list(SBM_seq10T100n50rho05, directed=TRUE, network_stats)
result_n50rho05[1,] <- colMeans(sim_result1) # apply(sim_result1, 2, median)


network_stats=c("edges", "mutual", "triangles")
sim_result2 <- CPD_STERGM_list(SBM_seq10T100n50rho05, directed=TRUE, network_stats)
result_n50rho05[2,] <- colMeans(sim_result2) # apply(sim_result2, 2, median)


sim_result3 <- Evaluation_gSeg(SBM_seq10T100n50rho05, p_threshold=0.05)
result_n50rho05[3,] <- colMeans(sim_result3)


sim_result4 <- Evaluation_kerSeg(SBM_seq10T100n50rho05, p_threshold=0.001)
result_n50rho05[4,] <- colMeans(sim_result4)


sim_result5 <- Evaluation_gSeg_on_stats(SBM_seq10T100n50rho05, p_threshold=0.05, num_stats=3)
result_n50rho05[5,] <- colMeans(sim_result5)


sim_result6 <- Evaluation_kerSeg_on_stats(SBM_seq10T100n50rho05, p_threshold=0.001, num_stats=3)
result_n50rho05[6,] <- colMeans(sim_result6)


write.csv(result_n50rho05, 'result_n50rho05.csv')


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



