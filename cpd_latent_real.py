import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import scipy.stats as st
torch.set_printoptions(precision=5)



def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument('--latent_dim', default=10, type=int) 
  parser.add_argument('--w_dim', default=5) 
  parser.add_argument('--output_layer', default=[32,64]) 
  parser.add_argument('--num_samples', default=200)
  parser.add_argument('--langevin_K', default=40)
  parser.add_argument('--langevin_s', default=0.5) 
  parser.add_argument('--kappa', default=10)
  parser.add_argument('--penalties', default=[10,20,50,100], type=int)
  parser.add_argument('--time_embed_dim', default=16, type=int)
  
  parser.add_argument('--epoch', default=100)
  parser.add_argument('--decoder_iteration', default=40)
  parser.add_argument('--nu_iteration', default=40)
  parser.add_argument('--decoder_lr', default=0.001, type=float)
  parser.add_argument('--decoder_thr', default=0.001)
  
  parser.add_argument('--iter_thr', default=5)
  parser.add_argument('--mu_thr', default=0.001)
  parser.add_argument('--loglik_thr', default=0.00001)

  parser.add_argument('--use_data', default='MITphone') # MITphone, Enron
  parser.add_argument('--directed', default=False) 
  parser.add_argument('--num_node', default=96, type=int) # MITphone: 96, Enron: 100
  parser.add_argument('--num_time', default=232, type=int) # MITphone: 232, Enron: 100
  parser.add_argument('--data_dir', default='./data/')
  parser.add_argument('-f', required=False)

  return parser.parse_args()




args = parse_args(); # print(args)
if torch.cuda.is_available():
  device = torch.device('cuda:0')
else:
  device = torch.device('cpu')
print('[INFO]', device)



# create folder
if args.use_data == 'MITphone':
  output_dir = os.path.join('MITphone')
  args.latent_dim = 10
  args.w_dim = 5
  args.epoch = 100
  args.decoder_lr = 0.001
  args.num_node = 96
  args.num_time = 232
  args.directed = False
elif args.use_data == 'Enron':
  output_dir = os.path.join('Enron')
  args.latent_dim = 5
  args.w_dim = 15
  args.epoch = 50
  args.decoder_lr = 0.01
  args.num_node = 100
  args.num_time = 100
  args.directed = False

os.makedirs(output_dir, exist_ok=True)



###################
# LOAD SAVED DATA #
###################

if args.use_data == 'MITphone':
  print('[INFO] MITphone')
  data = torch.from_numpy(np.load(args.data_dir + 'MITphone.npy'))
elif args.use_data == 'Enron':
  print('[INFO] Enron')
  data = torch.from_numpy(np.load(args.data_dir + 'Enron.npy'))


data = data.float()
print('[INFO] data loaded with dimension:', data.shape)
print('[INFO] directed:', args.directed)
print(args)


def init_weights(m):
  for name, param in m.named_parameters():
    nn.init.uniform_(param.data, -0.05, 0.05)



#########
# MODEL #
#########

class CPD(nn.Module):
  def __init__(self, args, half):
    super(CPD, self).__init__()

    self.d = args.latent_dim
    
    if half:
      self.T = int(args.num_time/2)
    else:
      self.T = args.num_time

    self.l1 = nn.Linear( self.d + self.d, args.output_layer[0])
    self.l2 = nn.Linear( args.output_layer[0] + self.d, args.output_layer[1])

    if args.directed:
      self.left = nn.Linear( args.output_layer[1] + self.d, args.num_node * args.w_dim, bias=False)
      self.right = nn.Linear( args.output_layer[1] + self.d, args.num_node * args.w_dim, bias=False)
    else:
      self.left = nn.Linear( args.output_layer[1] + self.d, args.num_node * args.w_dim, bias=False)
      
    self.t_embedding = nn.Linear( 1, self.d )

  def forward(self, z):
    # z: Tm by d

    time_points = torch.arange(0.0,self.T).repeat_interleave(args.num_samples).to(device)
    time_points_embed = self.t_embedding(time_points.unsqueeze(1))

    output = torch.cat([z, time_points_embed], dim=1)
    output = self.l1(output).tanh()
    output = torch.cat([output, time_points_embed], dim=1)
    output = self.l2(output).tanh()
    output = torch.cat([output, time_points_embed], dim=1)

    if args.directed:
      w_left   = self.left(output).tanh()   # Tm by nk
      w_right  = self.right(output).tanh()  # Tm by nk
      w_left   = w_left.reshape(self.T * args.num_samples, args.num_node, args.w_dim)   # Tm by n by k
      w_right  = w_right.reshape(self.T * args.num_samples, args.num_node, args.w_dim)  # Tm by n by k
      output   = torch.bmm(w_left, torch.transpose(w_right, 1, 2)).sigmoid() # Tm by n by n
    else:
      w_left   = self.left(output).tanh()   # Tm by nk
      w_left   = w_left.reshape(self.T * args.num_samples, args.num_node, args.w_dim)   # Tm by n by k
      output   = torch.bmm(w_left, torch.transpose(w_left, 1, 2)).sigmoid() # Tm by n by n
      
    return output
    
  def infer_z(self, z, adj_repeat, mu_repeat):
    # z: Tm by d
    # adj_train_repeat: Tm by n by n (repeated by m), with adj_train: T by n by n
    # mu_repeat: Tm by d (repeated by m)

    criterion = nn.BCELoss(reduction='sum') # negative log-likelihood for Bernoulli
    for k in range(args.langevin_K):
      z = z.detach().clone()
      z.requires_grad = True
      assert z.grad is None

      adj_pred = self.forward(z) # Tm by d -> Tm by n by n
      nll = criterion(adj_pred, adj_repeat) # both Tm by n by n
      z_grad_nll = torch.autograd.grad(nll, z)[0] # Tm by d
      noise = torch.randn(self.T * args.num_samples, self.d).to(device) # Tm by d

      # Langevin dynamics sampling
      z = z + torch.tensor(args.langevin_s) * (-z_grad_nll - (z - mu_repeat)) +\
          torch.sqrt(2 * torch.tensor(args.langevin_s)) * noise
          
    z = z.detach().clone() # Tm by d
    return z

  def cal_loglik(self, mu, adj_repeat):
    # mu: T by d
    # adj_repeat: Tm by n by n

    with torch.no_grad():

      # the constant
      loglik = -self.T * torch.log(torch.tensor(args.num_samples))
      loglik = loglik.to(device) 

      # create repeated version of mu, from (T by d) to (Tm by d)
      mu_repeat = np.repeat(mu.cpu().numpy(), np.repeat(args.num_samples, self.T), axis=0)
      mu_repeat = torch.from_numpy(mu_repeat).to(device)

      # this is the sample z drawn from marginal p(z) NOT FROM posterior p(z|y)
      z = mu_repeat + torch.randn(self.T * args.num_samples, self.d).to(device) 
      adj_prob = self.forward(z) # Tm by n by n

      # P(y_ij|z) = pk + (1-p)*(1-k)
      prob = adj_prob * adj_repeat + (1.0-adj_prob) * (1.0-adj_repeat) # element-wise operation: Tm by n by n
      log_prob = torch.log(prob).reshape(self.T, args.num_samples, args.num_node**2) # T by m by n*n

      sum_over_ij = torch.sum(log_prob, dim=2) # T by m
      max_over_m = torch.max(sum_over_ij, dim=1).values # T

      for t in range(self.T):
        sum_per_time = sum_over_ij[t,:] # m 
        sum_minus_max = sum_per_time - max_over_m[t] # m # subtract the max value from each of m elements
        sum_of_exp = torch.sum( torch.exp(sum_minus_max) ) # sum over m samples within time t
        loglik += max_over_m[t] + torch.log(sum_of_exp)

      loglik = loglik.detach().clone()
    
    return loglik 
    










def learn_one_seq_penalty(args, adj_train, adj_test, pen_iter, half):
  torch.manual_seed(1)
  
  m = args.num_samples
  kappa = args.kappa
  d = args.latent_dim
  penalty = args.penalties[pen_iter]
  early_stopping = False
  stopping_count = 0 # for ADMM
  epoch = args.epoch

  if half:
    T = int(args.num_time/2)
    label = 'half'
  else:
    T = args.num_time
    label = 'full'

  # create matrix X and vector 1
  ones_col = torch.ones(T, 1).to(device)
  X = torch.zeros(T, T-1).to(device)
  i, j = torch.tril_indices(T, T-1, offset=-1)
  X[i, j] = 1


  old_loglik = -float('inf')
  loglik_train_holder = []
  loglik_test_holder = []
  mu_diff_holder = []
  decoder_loss_holder = []
  CV_holder = []

  # use Coefficient of Variation (CV) when half=True
  # save result based on Coefficient of Variation (CV)
  if not half:
    best_mu = torch.zeros(T,d)
    best_loglik = torch.zeros(1)
    best_CV = -float('inf') # Coefficient of Variation
    best_CV_iter = 0

  
  # initialize mu, nu, w, with dim T by d of zeros
  mu = torch.zeros(T, d).to(device)
  nu = torch.zeros(T, d).to(device)
  w = torch.zeros(T, d).to(device)
  
  mu_old = mu.detach().clone()
  nu_old = nu.detach().clone()


  # creat repeated version of ground truth, from (T by n by n) to (Tm by n by n)
  # repeat m for T times, giving [m, m, ..., m]
  # for each t in T (axis=0), repeat num_samples times, giving (Tm by n by n)
  adj_train_repeat = np.repeat(adj_train.numpy(), np.repeat(m, T), axis=0) 
  adj_train_repeat = torch.from_numpy(adj_train_repeat).to(device)

  if half:
    # these objects exist if half = True
    adj_test_repeat = np.repeat(adj_test.numpy(), np.repeat(m, T), axis=0) 
    adj_test_repeat = torch.from_numpy(adj_test_repeat).to(device)


  model = CPD(args, half).to(device)
  model.apply(init_weights)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.decoder_lr)
  criterion = nn.BCELoss(reduction='sum') # loglik sum over t, but expectation over m, so later divided by m
  
  for learn_iter in range(epoch):

    ####################
    # GENERATE SAMPLES #
    ####################
    # create repeated version of mu, from (T by d) to (Tm by d)
    mu_repeat = np.repeat(mu.cpu().numpy(), np.repeat(m, T), axis=0)
    mu_repeat = torch.from_numpy(mu_repeat).to(device) # Tm by d
    init_z = torch.randn(T*m, d).to(device) # Tm by d, starts from N(0,1)
    sampled_z_all = model.infer_z(init_z, adj_train_repeat, mu_repeat) # Tm by d

    ################
    # UPDATE PRIOR # 
    ################
    expected_z = sampled_z_all.clone().reshape(T,m,d) # T by m by d
    expected_z = expected_z.mean(dim=1) # T by d
    mu = ( expected_z + kappa * (nu-w) ) / ( 1.0 + kappa )
    mu = mu.detach().clone()

    ##################
    # UPDATE DECODER #
    ##################
    inner_loss = float('inf')
    for decoder_iter in range(args.decoder_iteration):
      optimizer.zero_grad()
      adj_prob = model(sampled_z_all) # Tm by n by n 
      loss = criterion(adj_prob, adj_train_repeat) / m
      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), 1)
      optimizer.step()

      # early stopping
      loss_relative_diff = abs( (loss.item() - inner_loss) / inner_loss )
      inner_loss = loss.item()
      if loss_relative_diff < args.decoder_thr: 
        break

    #########################
    # UPDATE BETA AND GAMMA #
    #########################

    gamma = nu[0, :].unsqueeze(0) # row vector
    beta = torch.diff(nu, dim=0)

    for nu_iter in range(args.nu_iteration):
      # update beta once (t range from 1 to 99, NOT 1 to 100)
      for t in range(T-1): 
        beta_without_t = beta.detach().clone()
        X_without_t = X.detach().clone()
        beta_without_t[t,:] = torch.zeros(d) # make this row zeros
        X_without_t[:,t] = torch.zeros(T) # make this column zeros
        bt = kappa * torch.mm( X[:,t].unsqueeze(0), mu + w - torch.mm(ones_col, gamma) - torch.mm(X_without_t, beta_without_t) )
        bt_norm = torch.norm(bt, p=2)

        # UPDATE: soft-thresholding
        if bt_norm < penalty:
          beta[t,:] = torch.zeros(d)
        else:
          beta[t,:] = 1 / (kappa * torch.norm(X[:,t], p=2)**2) * (1 - penalty/bt_norm) * bt
        beta = beta.detach().clone()

      # check KKT
      counter_zero = 0
      counter_non_zero = 0
      for t in range(T-1):
        if torch.norm(beta[t,:], p=2) == 0.0:
          # beta is zeros
          condition = -kappa * torch.mm( X[:,t].unsqueeze(0), mu + w - torch.mm(ones_col, gamma) - torch.mm(X, beta) )
          if torch.norm(condition, p=2) <= penalty: 
            counter_zero += 1
        else:
          # beta is not zeros
          condition = -kappa * torch.mm( X[:,t].unsqueeze(0), mu + w - torch.mm(ones_col, gamma) - torch.mm(X, beta) )
          condition += penalty * beta[t,:] / torch.norm(beta[t,:], p=2)
          if torch.norm(condition, p=2) < 0.01:
            counter_non_zero += 1

      # update gamma
      gamma = torch.mean(mu + w - torch.mm(X, beta), dim=0).unsqueeze(0).detach().clone()

      if (counter_zero+counter_non_zero) == (T-1):
        break
        
    # recollect nu
    nu = torch.mm(ones_col, gamma) + torch.mm(X, beta)
    nu = nu.detach().clone()

    ############
    # UPDATE W # 
    ############

    w = mu - nu + w
    w = w.detach().clone()

    ############
    # RESIDUAL # 
    ############

    primal_residual = torch.sqrt(torch.mean(torch.square(mu - nu)))
    dual_residual = torch.sqrt(torch.mean(torch.square(nu - nu_old)))

    if primal_residual > 10.0 * dual_residual:
      kappa *= 2.0
      w *= 0.5
      print('\n[INFO] kappa increased to', kappa)
    elif dual_residual > 10.0 * primal_residual:
      kappa *= 0.5
      w *= 2.0
      print('\n[INFO] kappa decreased to', kappa)

    
    # calculate log_likelihood
    with torch.no_grad():
      # if half=False, adj_test_repeat does not exist
      if half:
        loglik_train = model.cal_loglik(mu, adj_train_repeat) # USE TRAIN
        loglik_train_holder.append(loglik_train.detach().cpu().numpy().item())
        loglik = model.cal_loglik(mu, adj_test_repeat) # USE TEST
        loglik_test_holder.append(loglik.detach().cpu().numpy().item())
      else:
        loglik = model.cal_loglik(mu, adj_train_repeat) # USE TRAIN
        loglik_train_holder.append(loglik.detach().cpu().numpy().item())
      
      # criteria 1
      loglik_relative_diff = torch.abs((loglik - old_loglik) / old_loglik)
      old_loglik = loglik.detach().clone()

      # criteria 2
      mu_relative_diff = torch.norm(mu-mu_old, p='fro')
      mu_diff_holder.append(mu_relative_diff.detach().cpu().numpy().item())

      # decoder loss
      #decoder_loss_holder.append(loss.item())
      
      mu_old = mu.detach().clone()
      nu_old = nu.detach().clone()

    #####################
    # STOPPING CRITERIA #
    #####################
 
    if loglik_relative_diff < args.loglik_thr:
      stopping_count += 1
    else:
      stopping_count = 0

    if stopping_count >= args.iter_thr:
      print('\n[INFO] early stopping')
      early_stopping = True

    #########################
    # SAVE BEST RESULT (CV) #
    #########################
    # Coefficient of Variation (CV)
    if not half:
      delta_mu = torch.norm(torch.diff(mu, dim=0), p=2, dim=1).detach().clone()
      CV = torch.std(delta_mu)/torch.mean(delta_mu) # Coefficient of Variation (CV)
      CV_holder.append(CV.detach().cpu().numpy().item())
      if CV > best_CV and learn_iter > 10:
        best_mu = mu.detach().clone()
        best_loglik = loglik.detach().clone()
        best_CV = CV.detach().clone()
        best_CV_iter = learn_iter

    ##############
    # PRINT INFO #
    ##############

    if (learn_iter+1) % 10 == 0:

      with torch.no_grad():

        # second row - first row
        delta_mu = torch.norm(torch.diff(mu, dim=0), p=2, dim=1)
        delta_mu = delta_mu.cpu().detach().numpy() # numpy for plot

        plt.plot(delta_mu)
        plt.savefig( output_dir + '/{}_delta_mu_pen{}_learn{}'.format(label,pen_iter,learn_iter+1) + '.png' ) 
        plt.close()

        if half: 
          plt.plot(loglik_train_holder[1:], label="train")
          plt.plot(loglik_test_holder[1:], label="test")
          plt.legend(loc="lower right")
          plt.savefig( output_dir + '/{}_loglik_pen{}'.format(label,pen_iter) + '.png' ) 
          plt.close()
        else:
          plt.plot(loglik_train_holder[1:])
          plt.savefig( output_dir + '/{}_loglik_pen{}'.format(label,pen_iter) + '.png' ) 
          plt.close()

        plt.plot(mu_diff_holder[1:])
        plt.savefig( output_dir + '/{}_mu_diff_pen{}'.format(label,pen_iter) + '.png' ) 
        plt.close()

        if not half:
          plt.plot(CV_holder)
          plt.savefig( output_dir + '/{}_CV_pen{}'.format(label,pen_iter) + '.png' ) 
          plt.close()

        print('\nlearning iter ([penalty={}], data={}) ='.format(penalty,label), learn_iter+1, 'of', epoch)
        print('\tlog likelihood =', loglik)
        print('\tprimal residual =', primal_residual)
        print('\tdual residual =', dual_residual)
        print('\t\tlog likelihood relative difference =', loglik_relative_diff)
        print('\t\tmu relative difference =', mu_relative_diff)
      
    ###############
    # SAVE RESULT #
    ###############
    # at the last iteration or early_stopping
    if (learn_iter+1) == epoch or early_stopping:
      print('\nFINAL learning iter ([penalty={}]) ='.format(penalty), learn_iter+1, 'of', epoch)
      print('FINAL log likelihood =', loglik)
      print('FINAL log likelihood relative difference =', loglik_relative_diff)
      print('FINAL mu relative difference =', mu_relative_diff)

      with torch.no_grad():
        if half:
          # USE THE LAST MU
          delta_mu = torch.norm(torch.diff(mu, dim=0), p=2, dim=1)
          return loglik, mu
        else:
          # USE THE BEST MU
          print('[INFO] best_CV_iter =', best_CV_iter)
          delta_mu = torch.norm(torch.diff(best_mu, dim=0), p=2, dim=1)
          return best_loglik, best_mu

      

######################
# parameter learning #
######################



best_loglik = torch.tensor(-float('inf')) # keep updated throughout cross-validation


identity_matrices = torch.eye(args.num_node).unsqueeze(0).repeat(args.num_time, 1, 1)
one_seq = data # one sequence (T by n by n)
one_seq = one_seq + identity_matrices # add self edges

# visualization of edge count
sums = one_seq.reshape(args.num_time, -1).sum(dim=1)
plt.plot(list(range(1, args.num_time + 1)), sums)
plt.title('Edge Count over Time')
plt.savefig(output_dir + '/edge_count.png')
plt.close()


##############
# SPLIT DATA #
##############
odd_idx = range(1, args.num_time, 2)
even_idx = range(0, args.num_time, 2)
train_data = one_seq[odd_idx,:,:]
test_data = one_seq[even_idx,:,:]
print('[INFO] train_data.shape:', train_data.shape)
print('[INFO] test_data.shape:', test_data.shape)

######################
# TRAIN ON HALF DATA #
######################
holder_loglik = []
holder_index_comb = []

# same sequence, different penalties, different variance
for pen_iter in range(len(args.penalties)):
  test_loglik, _ = learn_one_seq_penalty(args, train_data, test_data, pen_iter, half=True)
  holder_loglik.append(test_loglik.cpu().numpy().item())
  holder_index_comb.append(pen_iter)
  print('holder_loglik:\n',holder_loglik)
  print('holder_index_comb:\n',holder_index_comb)

  if test_loglik > best_loglik:
    print('[INFO] best_loglik is updated')
    best_loglik = test_loglik

# model selection via Cross Validation
best_comb_index = holder_loglik.index(max(holder_loglik))
best_comb = holder_index_comb[best_comb_index]
pen_selection = best_comb
print('[INFO] max test loglik at pen_iter={}'.format(pen_selection))

#####################
# TEST ON FULL DATA #
#####################

# use one_seq and pen_iter_selection with 'test_data = None'
_, mu_output = learn_one_seq_penalty(args, one_seq, None, pen_selection, half=False) 

# second row - first row
delta_mu = torch.norm(torch.diff(mu_output, dim=0), p=2, dim=1)
delta_mu = delta_mu.cpu()


t_change = (delta_mu - torch.median(delta_mu)) / torch.std(delta_mu)
threshold = torch.mean(t_change) + torch.tensor(st.norm.ppf(0.9)) * torch.std(t_change)

# store change points (exclude first and last 5)
est_CP = []
tau = args.num_time - 1

for i in range(tau):
  if t_change[i] > threshold and i >= 5 and i <= tau-5:
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
    to_remove = selection[torch.argmin(delta_mu[selection])] # keep the highest
    est_CP.remove(to_remove)


# there are 100 time points, 99 differences
# if the 25th difference (index 24) > threshold, the actual time point is 26, so index 24+2=26
est_CP = [cp + 2 for cp in est_CP] # CP as actual time point

# save final visualization
plt.plot(np.arange(args.num_time-1), t_change)
plt.axhline(y=threshold.numpy(), color='r', linestyle='-')
plt.savefig( output_dir + '/mu_CP_pen{}.png'.format(pen_selection) ) 
plt.close()

torch.save(mu_output, os.path.join(output_dir, 'mu_par.pt') )
np.save('mu_par_{}.npy'.format(args.use_data), mu_output.cpu().numpy())
















