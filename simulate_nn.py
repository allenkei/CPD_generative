import argparse
import torch
import torch.nn as nn
import numpy as np
import os
torch.manual_seed(1)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--scaling', default=1.0, type=float)
  parser.add_argument('--latent_dim', default=10) 
  parser.add_argument('--w_dim', default=5) 
  parser.add_argument('--num_seq', default=10)
  parser.add_argument('--num_time', default=100)
  parser.add_argument('--num_node', default=50, type=int)
  parser.add_argument('--data_dir', default='./simulated_data/')
  parser.add_argument('-f', required=False) 
  return parser.parse_args()


args = parse_args(); print(args)


class simulate_adj(nn.Module):
  def __init__(self, args):
    super(simulate_adj, self).__init__()

    self.rnn_left = nn.RNN(args.latent_dim, args.num_node*args.w_dim, num_layers=1, batch_first=True)
    self.rnn_right = nn.RNN(args.latent_dim, args.num_node*args.w_dim, num_layers=1, batch_first=True)
    
    # Initialize parameters
    self.init_rnn_weights(self.rnn_left)
    self.init_rnn_weights(self.rnn_right)


  def init_rnn_weights(self, rnn):
    for name, param in rnn.named_parameters():
      if 'weight' in name:
        nn.init.uniform_(param, -0.5, 0.5)
      elif 'bias' in name:
        nn.init.uniform_(param, -0.5, 0.5)

  def forward(self, z):

    output_left, _ = self.rnn_left(z) # (batch_size, seq_length, hidden_size)
    output_right, _ = self.rnn_right(z) # (batch_size, seq_length, hidden_size)

    output_left = output_left.squeeze(0) # (seq_length, hidden_size)
    output_right = output_right.squeeze(0) # (seq_length, hidden_size)

    output_left = output_left.reshape(args.num_time, args.num_node, args.w_dim)
    output_right = output_right.reshape(args.num_time, args.num_node, args.w_dim)

    output = torch.bmm(output_left, torch.transpose(output_right, 1, 2)).sigmoid() # T by n by n


    return output



# save model parameters
model = simulate_adj(args)
checkpoint_path = os.path.join(args.data_dir, "model_par_n{}.pth".format(args.num_node))
torch.save(model.state_dict(), checkpoint_path)
model.eval()



adj_sequence = torch.zeros(args.num_seq, args.num_time, args.num_node, args.num_node)
additions = [
    torch.full((25, args.latent_dim), -1.0),  # t = 0 to 24
    torch.full((25, args.latent_dim), 5.0),  # t = 25 to 49
    torch.full((25, args.latent_dim), -1.0),  # t = 50 to 74
    torch.full((25, args.latent_dim), 5.0)   # t = 75 to 99
]
additions = torch.cat(additions, dim=0)

#print(additions)



for gen_iter in range(args.num_seq):
  z = torch.empty(args.num_time, args.latent_dim).normal_(mean=0,std=0.1) 
  z = z + additions
  #print('z:\n',z)
  
  z = z.unsqueeze(0) 

  adjacency_matrices = model(z) # T by n by n
  print('adjacency_matrices.shape:',adjacency_matrices.shape)

  adjacency_matrices *= args.scaling
  adjacency_matrices = torch.bernoulli(adjacency_matrices) # T by n by n

  for t in range(args.num_time):
    adjacency_matrices[t,:,:].fill_diagonal_(0) # make sure diagonal is all 0

  adj_sequence[gen_iter,:,:,:] = adjacency_matrices


adj_sequence = adj_sequence.detach().clone()
print('dimension of generated data:', adj_sequence.shape)

print(adj_sequence[0,0,0:15,0:15])
print(adj_sequence[0,25,0:15,0:15])
print(adj_sequence[0,50,0:15,0:15])



adj_sequence = adj_sequence.numpy()
adj_sequence_path = os.path.join(args.data_dir, "NN_seq{}T{}n{}.npy".format(args.num_seq,args.num_time,args.num_node))
np.save(adj_sequence_path, adj_sequence)







