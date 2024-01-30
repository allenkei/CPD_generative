
library(devtools)
install_github("allenkei/CPDstergm")
library(CPDstergm)
source("EVAL.R") # FILE DIRECTORY

load("STERGM_seq10T100n50p6.RData") # DATA DIRECTORY
result_n50p6 <- matrix(0, nrow=6, ncol=4)


network_stats=c("edges", "mutual")
sim_result1 <- CPD_STERGM_list(STERGM_seq10T100n50p6, directed=TRUE, network_stats)
result_n50p6[1,] <- colMeans(sim_result1)


network_stats=c("edges", "mutual", "triangles")
sim_result2 <- CPD_STERGM_list(STERGM_seq10T100n50p6, directed=TRUE, network_stats)
result_n50p6[2,] <- colMeans(sim_result2)


sim_result3 <- Evaluation_gSeg(STERGM_seq10T100n50p6, p_threshold=0.05)
result_n50p6[3,] <- colMeans(sim_result3)


sim_result4 <- Evaluation_kerSeg(STERGM_seq10T100n50p6, p_threshold=0.001)
result_n50p6[4,] <- colMeans(sim_result4)


sim_result5 <- Evaluation_gSeg_on_stats(STERGM_seq10T100n50p6, p_threshold=0.05, num_stats=3)
result_n50p6[5,] <- colMeans(sim_result5)


sim_result6 <- Evaluation_kerSeg_on_stats(STERGM_seq10T100n50p6, p_threshold=0.001, num_stats=3)
result_n50p6[6,] <- colMeans(sim_result6)


write.csv(result_n50p6, 'result_n50p6.csv')


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

