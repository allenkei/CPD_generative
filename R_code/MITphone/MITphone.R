library(CPDstergm)
library(reticulate)

data("MITphone") # DATA FROM library(CPDstergm)


save_to_numpy_MITphone <- function(data_list, file_name){
  num_time <- length(data_list)
  num_node <- dim(data_list[[1]])[1] # first dim of matrix
  
  output <- array(NA, dim = c(num_time, num_node, num_node))
  
  for(time_iter in 1:num_time){
    output[time_iter,,] <- data_list[[time_iter]]
  }
  
  np = import("numpy")
  np$save(file_name, r_to_py(output))
  print('File saved')
}

save_to_numpy_MITphone(MITphone, "MITphone.npy") # Python



library(CPDstergm)
source("EVAL.R") # FILE DIRECTORY

data("MITphone")
result <- CPD_STERGM(MITphone, directed=FALSE, network_stats=c("edges", "isolates", "triangles"))


gSeg_result <- Evaluation_gSeg_on_stats(MITphone, p_threshold=0.05, num_stats=3, is_experiment=TRUE)
kerSeg_result <- Evaluation_kerSeg_on_stats(MITphone, p_threshold=0.001, num_stats=3, is_experiment=TRUE)


theta_change <- result$theta_change; threshold <- result$threshold; xtick <- result$est_CP
seq_date <- seq(as.Date("2004-09-15"), as.Date("2005-05-04"), by="days"); tau <- length(seq_date)-1








library(reticulate)
np <- import("numpy")
mu_output <- np$load("mu_par_MIT.npy") # PARAMETER DIRECTORY

# calculate sequential differences
tau <- dim(mu_output)[1] - 1
delta_mu <- numeric(tau)
for(i in 1:(tau-1)){delta_mu[i] <- norm(mu_output[i+1,]-mu_output[i,],"2")}

# calculate threshold
med <- median(delta_mu); std <- sd(delta_mu)
t_change <- (delta_mu - med)/std # normalize
threshold <- mean(t_change) + qnorm( 1-0.1, lower.tail = T) * sd(t_change)
delta_mu <- t_change

# calculate CP
est_CP <- c()
for(i in 1:tau){
  if(delta_mu[i] > threshold & i > 10 & i < (tau-10)) {
    est_CP <- c(est_CP, i) # location of change point
  }
}

#min-spacing
end_i <- 2
while(end_i <= length(est_CP)){
  prev <- est_CP[end_i-1]
  this <- est_CP[end_i]
  
  if(this - prev > 10){
    end_i <- end_i + 1
  }else{
    selection <- c(prev,this)
    to_remove <- selection[which.min(delta_mu[selection])]
    est_CP <- est_CP[-which(est_CP == to_remove)]
  }
};rm(np, med, std, prev, this, to_remove, selection, end_i, i, t_change)


est_CP <- est_CP + 1 # the 1st delta_mu indicates t=2 is change point




#################
# VISUALIZATION #
#################
# 8 by 5

par(mar=c(2, 4, 2, 1), fig=c(0,1,0,0.82))
plot(1:length(delta_mu), delta_mu, type='l',ylab="", xlab="", xaxt="n", yaxt="n")
abline(h = threshold, col='red',lwd=2)
seq_date <- seq(as.Date("2004-09-15"), as.Date("2005-05-04"), by="days")
xtick <- est_CP-1 # the xtick is for delta_mu, so minus 1
axis(side=1, at=xtick, labels = F, lwd = 0, lwd.ticks = 1) # est_CP is actual time
text(x=xtick,  par("usr")[3]-1, labels = seq_date[est_CP], cex=0.8, xpd=TRUE)

ytick <- c(0,2,4,6,8)
axis(side=2, at=ytick, labels = FALSE)
text(par("usr")[1]-1.7, ytick, labels=ytick, pos=2, xpd=TRUE, cex=0.8)


par(mar=c(0, 4, 0, 1), fig=c(0,1,0.75,0.82), new=T)
plot(NULL, ylim=c(0,1), xlim=c(1,tau), ylab="", xlab="", xaxt="n", yaxt="n")
for(i in result$est_CP){abline(v=i-1, col='blue', lwd=2)}
text(par("usr")[1]+1, 0.45, labels='CPDstergm', pos=2, xpd=TRUE, cex=0.8)

par(mar=c(0, 4, 0, 1), fig=c(0,1,0.83,0.9), new=T)
plot(NULL, ylim=c(0,1), xlim=c(1,tau), ylab="", xlab="", xaxt="n", yaxt="n")
for(i in kerSeg_result){abline(v=i-1, col='blue', lwd=2)}
text(par("usr")[1]+1, 0.45, labels='kerSeg', pos=2, xpd=TRUE, cex=0.8)

par(mar=c(0, 4, 0, 1), fig=c(0,1,0.91,0.98), new=T)
plot(NULL, ylim=c(0,1), xlim=c(1,tau), ylab="", xlab="", xaxt="n", yaxt="n")
for(i in gSeg_result){abline(v=i-1, col='blue', lwd=2)}
text(par("usr")[1]+1, 0.45, labels='gSeg', pos=2, xpd=TRUE, cex=0.8)





