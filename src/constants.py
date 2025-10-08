import torch 

# define device
CUDA = 'cuda'
CPU = 'cpu'
DEVICE = torch.device("mps" if torch.mps.is_available() else CPU)
SEED = 1234


