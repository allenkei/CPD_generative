#install.packages('igraphdata')
#install.packages('igraph')
library(igraphdata)
library(igraph)
library(dplyr)
library(reticulate)


data("enron", package = "igraphdata")
network_igraph <- upgrade_graph(enron) # network_igraph <- enron
rm(enron)

network_edgelist <- as.data.frame(as_edgelist(network_igraph))
network_edgelist <- cbind(network_edgelist, Time = E(network_igraph)$Time)




# remove data before 1990
removal <- which(network_edgelist$Time < as.Date("1990-01-01"))
network_edgelist <- network_edgelist[-removal,]
rm(network_igraph, removal)


# check time range
#min(network_edgelist$Time)
#max(network_edgelist$Time)



start_date <- end_date <- as.Date("2000-06-05")
num_of_week <- 100

for(time_iter in 1:num_of_week){
  end_date <- end_date + 7
}; rm(time_iter)



# get frequent user
network_edgelist <- network_edgelist[which(network_edgelist$Time >= start_date & 
                                             network_edgelist$Time < end_date),]


all_users <- c(network_edgelist$V1, network_edgelist$V2)
all_users <- table(all_users) %>% as.data.frame() %>% arrange(desc(Freq)) # sorting
top_users <- all_users[1:100,] # select top 100 frequent user
selected_users <- droplevels(top_users$all_users)
rm(all_users, top_users, end_date)


y_list <- list()
edge_sum <- c()


for(time_iter in 1:num_of_week){
  end_date <- start_date + 7
  
  y <- matrix(0, nrow=100, ncol=100)
  temp <- network_edgelist[which(network_edgelist$Time >= start_date & network_edgelist$Time < end_date),]
  
  for(iter in 1:dim(temp)[1]){
    node_i <- which(temp[iter,1] == selected_users) # index out of 100
    node_j <- which(temp[iter,2] == selected_users) # index out of 100
    
    # both exist in selected_users
    if(length(node_i) == 1 & length(node_j) == 1){ 
      # for undirected
      if(y[node_i, node_j] == 0 & node_i != node_j){y[node_i, node_j] <- 1}
      if(y[node_j, node_i] == 0 & node_i != node_j){y[node_j, node_i] <- 1}
    }
  }; rm(node_i, node_j, temp)
  
  y_list[[time_iter]] <- y
  edge_sum <- c(edge_sum, sum(y))
  start_date <- end_date
}

rm(iter, time_iter, start_date, end_date, y, selected_users, network_edgelist)

# plot edge count
plot(1:num_of_week, edge_sum, type='l')

save_to_numpy <- function(data_list, file_name){
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


save_to_numpy(y_list, "Enron.npy") # Python




###############
# EXPERIMENTS #
###############

source("EVAL.R") # FILE DIRECTORY

library(CPDstergm)
result <- CPD_STERGM(y_list, directed=FALSE, network_stats=c("edges", "isolates", "triangles"))


gSeg_result <- Evaluation_gSeg_on_stats(y_list, p_threshold=0.05, num_stats=3, is_experiment=TRUE)
kerSeg_result <- Evaluation_kerSeg_on_stats(y_list, p_threshold=0.001, num_stats=3, is_experiment=TRUE)


theta_change <- result$theta_change; threshold <- result$threshold; xtick <- result$est_CP
seq_date <- seq(as.Date("2000-06-05"), as.Date("2002-05-06"), by="weeks")
seq_date <- seq_date[1:100] # remove the last one




library(reticulate)
np <- import("numpy")
mu_output <- np$load("mu_par_Enron.npy") # PARAMETER DIRECTORY

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
  
  if(this - prev >= 10){
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
xtick <- est_CP-1 # the xtick is for delta_mu, so minus 1
axis(side=1, at=xtick, labels = F, lwd = 0, lwd.ticks = 1) # est_CP is actual time
for(i in 1:3){ # There are four change points
  text(x=xtick[i]-3,  par("usr")[3]-0.5, labels = seq_date[est_CP[i]], cex=0.8, xpd=TRUE)
}
text(x=xtick[4]+3,  par("usr")[3]-0.5, labels = seq_date[est_CP[4]], cex=0.8, xpd=TRUE) # The last one

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







