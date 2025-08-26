import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler

#readme
#Define MPLayer_in_opt - TEMP based  Layer 
#Define MPLayer_in_K - pi^2_K Layer 


class MPLayer_in_K(torch.nn.Module):
  def __init__(self,inp_node,out_node,gamma,diff=0,sparse=0,drop_prob=0.5):
    super().__init__()
    self.inp_node = inp_node
    self.out_node = out_node
    self.gamma = gamma
    self.diff = diff # differential inputs are given or not
    # torch.manual_seed(43)
    self.sparse = sparse
    self.drop_prob = drop_prob
    self.weight = torch.nn.Parameter(torch.empty(inp_node, out_node), requires_grad=True)
    torch.nn.init.xavier_normal_(self.weight, gain=1.0)
    torch.clamp(self.weight,-3,3)

  def spikeK(self, sorted_in: torch.Tensor, gamma: float):
      if gamma == 0 or gamma == 1:
          out = torch.kthvalue(sorted_in, 1, dim=1).values
          return out
      thr,_ = torch.topk(sorted_in, gamma, dim=1, largest=False, sorted=False)
      sum_nonzero = thr.sum(dim=1)
      return (sum_nonzero/(gamma)) #avg of min K values

  def forward(self, inputp, inputn=None):
      inputp = torch.unsqueeze(inputp,axis=-1)
      self.weight.type_as(inputp)
      if(inputn==None):
        plusIn = F.relu((3+inputp))
        minusIn = F.relu((3-inputp))
      else:
        minusIn = torch.unsqueeze(inputn,axis=-1)
        plusIn = inputp
      
      plusW = F.relu(3+self.weight)
      minusW = F.relu(3-self.weight)
       
      if(self.sparse):
        rand_idx = torch.randperm(self.inp_node)
        l = torch.round(torch.tensor(self.drop_prob*self.inp_node)).item()
        rand_idx = rand_idx[:int(l)]
        plusIn = plusIn[:, rand_idx, :]
        minusIn = minusIn[:, rand_idx, :]
        plusW = (plusW[rand_idx,:])
        minusW = (minusW[rand_idx,:])
        
      zPlus = torch.cat([(plusIn+plusW),(minusIn+minusW)],axis=1)
      zMinus = torch.cat([(plusIn+minusW),(minusIn+plusW)],axis=1)
      
      zPlus = self.spikeK(zPlus, self.gamma)
      zMinus = self.spikeK(zMinus, self.gamma)
      torch.cuda.empty_cache()
      if(self.diff == 0):
        return zPlus - zMinus  ## previous TEMP based codes will not be compatible because of this change
      else:
        return zPlus,zMinus
