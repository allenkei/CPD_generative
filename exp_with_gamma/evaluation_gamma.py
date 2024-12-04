import torch
import numpy as np
from math import comb
import scipy.stats as st
import matplotlib.pyplot as plt

def evaluation_gamma(mu, delta_mu, args, loglik, pen_iter, seq_iter, output_dir, half):

  # mu: T by d
  # delta_mu: T-1 norm2

  torch.manual_seed(1)
  np.random.seed(1)

  alpha = 0.01 # for threshold
  false_alarm_tolerance = 2.0 # for false alarm ratio (within tolerance window)
  if args.latent_dim == 5:
    num_sample = args.gamma_num_samples * 2
  elif args.latent_dim == 10:
    num_sample = args.gamma_num_samples 
  
  if half:
    T = int(args.num_time/2)
    tau = T-1 # 50 time points - 1 = 49 differences
    true_CP = args.true_CP_half # [14,26,39]
    label = 'half'
  else:
    T = args.num_time
    tau = T-1 # 100 time points - 1 = 99 differences
    true_CP = args.true_CP_full # [26,51,76]
    label = 'full'

  
  mu = mu.cpu() # T by d
  delta_mu_matrix = np.diff(mu, axis=0) # T-1 by d
  delta_mu = delta_mu.cpu() # T-1

  # threshold from gamma distribution
  threshold = st.gamma.ppf(1 - alpha/(T-1), a = num_sample*args.latent_dim/2, scale = 2/num_sample)

  half_z_norm2 = np.zeros(T-1) # T-1
  for time_idx in range(T-1):
    
    sampled_z = np.zeros((num_sample, args.latent_dim)) # n by d
    mean_vec = delta_mu_matrix[time_idx,:]

    for i in range(num_sample):
      # sample from N(mu^{t} - mu^{t-1}, 2I) # covariance is 2 * identity
      sampled_z[i,:] = np.random.multivariate_normal(mean_vec, 2*np.eye(args.latent_dim))

    sampled_z_norm2 = 0.5*np.linalg.norm(sampled_z, axis=1)**2 # n by 1
    half_z_norm2[time_idx] = np.mean(sampled_z_norm2) # average of the l2 norm for a time_idx
  
  # store change points (exclude first and last 5)
  est_CP = []
  for i in range(tau):
    if half_z_norm2[i] > threshold and i >= 5 and i <= tau-5:
      est_CP.append(i)

  # min-spacing
  end_i = 1
  while end_i < len(est_CP):
    prev = est_CP[end_i-1]
    this = est_CP[end_i]

    if this - prev > 5: # spacing > 5
      end_i += 1
    else:
      selection = [prev, this]
      to_remove = selection[torch.argmin(delta_mu[selection])] # keep the highest delta_mu
      est_CP.remove(to_remove)

  # there are 100 time points, 99 differences
  # if the 25th difference (index 24) > threshold, the actual time point is 26, so index 24+2=26
  est_CP = [cp + 2 for cp in est_CP] # CP as actual time point
  num_CP = len(est_CP)

  # intervals
  gt_CP_all = [1] + true_CP + [T + 1]
  est_CP_all = [1] + est_CP + [T + 1]
  gt_list = [range(gt_CP_all[i-1], gt_CP_all[i]) for i in range(1, len(gt_CP_all))]
  est_list = [range(est_CP_all[i-1], est_CP_all[i]) for i in range(1, len(est_CP_all))]

  if num_CP == 0:
    dist_est_gt = float('inf')
    dist_gt_est = float('-inf')
    covering_metric = 0
  else:
    # calculate the 2 one-sided distance
    holder_est_gt = []
    for i in true_CP:
      dist_diff_est_gt = [abs(j-i) for j in est_CP]
      holder_est_gt.append(min(dist_diff_est_gt))
    dist_est_gt = max(holder_est_gt)

    holder_gt_est = []
    for i in est_CP:
      dist_diff_gt_est = [abs(j-i) for j in true_CP]
      holder_gt_est.append(min(dist_diff_gt_est))
    dist_gt_est = max(holder_gt_est)

    # calculate covering metric
    covering_metric = 0
    for A in gt_list:
      jaccard = []
      for A_prime in est_list:
        jaccard.append( len(set(A).intersection(set(A_prime))) / len(set(A).union(set(A_prime))) )
      covering_metric += len(A) * max(jaccard)
    covering_metric /= tau + 1

  abs_error = abs(num_CP - len(true_CP))

  # calculate False-alarm ratio
  FP = 0.0
  deno = float(T - len(true_CP))

  for true_cp in true_CP:
    is_FP = True
    for detected_cp in est_CP:
      if abs(detected_cp - true_cp) <= false_alarm_tolerance:
        is_FP = False
        break
    if is_FP:
      FP += 1.0

  false_alarm = FP/deno

  # save final visualization
  plt.plot(np.arange(T-1), delta_mu); plt.xticks(true_CP)
  plt.savefig( output_dir + '/{}_best_delta_mu_seq{}pen{}.png'.format(label,seq_iter,pen_iter) ) 
  plt.close()

  plt.plot(np.arange(T-1), half_z_norm2); plt.xticks(true_CP)
  plt.axhline(y=threshold, color='r', linestyle='-')
  plt.savefig( output_dir + '/{}_CP_sampled_Z_seq{}pen{}.png'.format(label,seq_iter,pen_iter) ) 
  plt.close()
  
  output = [abs_error, dist_est_gt, dist_gt_est, covering_metric, false_alarm, loglik, threshold, est_CP]
  return output








