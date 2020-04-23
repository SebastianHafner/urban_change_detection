import torch

def to_numpy(tensor:torch.Tensor):
    return tensor.cpu().detach().numpy()

