import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_random_elements(tensor, drop_prob=0.3):
    """Randomly drops elements from a tensor with probability `drop_prob`."""
    mask = torch.rand(tensor.shape) > drop_prob  # Create a random mask
    return tensor[mask]

class K_layer_opt(nn.Module):
    def __init__(self, input_dim, output_dim, gamma, diff,sparse,drop_prob=0.3):
        super().__init__()
        self.gamma = gamma
        self.diff = diff
        self.sparse = sparse
        self.drop_prob = drop_prob
        # torch.manual_seed(45)
        self.weight = torch.nn.Parameter(torch.empty(input_dim, output_dim), requires_grad=True)
        torch.nn.init.xavier_normal_(self.weight, gain=1.0)
        self.weight.data = torch.clamp(self.weight.data, -3, 3)  # In-place clamping of weight values    
    
    def spikeK(self, sorted_in: torch.Tensor, gamma: float):
      if gamma == 0 or gamma == 1:
          out = torch.kthvalue(sorted_in, 1, dim=2).values
          return out
      thr,_ = torch.topk(sorted_in, gamma, dim=2, largest=False, sorted=False)
      sum_nonzero = thr.sum(dim=2)
      return (sum_nonzero/(gamma)) #avg of min K values

    def forward(self, input, inputn=None):
        input = torch.unsqueeze(input,axis=-1)
        if inputn is not None:
            inputp = input
            inputn = torch.unsqueeze(inputn, axis=-1)
        else:
            inputp = F.relu((3 + input)) #remove F.relu
            inputn = F.relu((3 - input))

        Wp = F.relu(3+self.weight)
        Wn = F.relu(3-self.weight)
        
        if(self.sparse):  
          a = inputp.size()[2]
          rand_idx = torch.randperm(a)
          drop_count = int(torch.round(torch.tensor(self.drop_prob * a)).item())
          rand_idx = rand_idx[:drop_count]
          inputp = inputp[:,:, rand_idx, :]
          inputn = inputn[:,:, rand_idx, :]
          Wp = (Wp[rand_idx,:])
          Wn = (Wn[rand_idx,:])
                  
        if(self.gamma > 0):
          zpp,_ = torch.topk((inputp + Wp), self.gamma, dim=2, largest=False, sorted=False)
          znp,_ = torch.topk((inputn + Wn), self.gamma, dim=2, largest=False, sorted=False)
          zpn,_ = torch.topk((inputp + Wn), self.gamma, dim=2, largest=False, sorted=False)
          znn,_ = torch.topk((inputn + Wp), self.gamma, dim=2, largest=False, sorted=False)
          del _
          zP = torch.cat([zpp,znp], axis=2)
          zN = torch.cat([zpn,znn], axis=2)        
        else:
          zP = torch.cat([(inputp + Wp),(inputn + Wn)], axis=2)
          zN = torch.cat([(inputn + Wp),(inputp + Wn)], axis=2)       
        tzP = self.spikeK(zP, self.gamma)
        tzN = self.spikeK(zN, self.gamma)        
        return tzP, tzN
 
 
class temp_conv_k_opt(nn.Module):
    def __init__(self, in_channels, out_channels, gamma,  kernel_size=3, dilation=1, padding=1, stride=1, diff=0,sparse = False,drop_prob=0.3):
        super(temp_conv_k_opt, self).__init__() 
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = (dilation, dilation)
        self.padding = padding
        self.stride = stride
        self.gamma = gamma
        self.diff = diff
        self.cnn = K_layer_opt(self.in_channels * self.kernel_size * self.kernel_size, self.out_channels, self.gamma, diff=self.diff,sparse=sparse,drop_prob=drop_prob)        
    
    def forward(self, inputs,inputn=None): 
        inp_size = inputs.size()     
        inp_unf_p = F.unfold(inputs, (self.kernel_size, self.kernel_size), stride=self.stride, padding=self.padding)
        patches = inp_unf_p.permute(0, 2, 1)  # Shape: [batch_size, num_patches, patch_size]
        if(inputn is not None):
          inp_unf_n = F.unfold(inputn, (self.kernel_size, self.kernel_size), stride=self.stride, padding=self.padding)
          patches_n = inp_unf_n.permute(0, 2, 1)  # Shape: [batch_size, num_patches, patch_size]
          output_p, output_n = self.cnn(patches,patches_n)
        else:
          output_p, output_n = self.cnn(patches)
        out_height = math.floor((inp_size[2] - self.kernel_size + 2 * self.padding) / self.stride) + 1 #compute new output height
        out_width = math.floor((inp_size[3] - self.kernel_size + 2 * self.padding) / self.stride) + 1 #compute new output width
        output_p = torch.reshape(output_p, (inp_size[0], out_height, out_width, self.out_channels))
        output_p = output_p.permute(0, 3, 1, 2)
        output_n = torch.reshape(output_n, (inp_size[0], out_height, out_width, self.out_channels))
        output_n = output_n.permute(0, 3, 1, 2)
        if(self.diff==0):    ##changes made here
          return output_p - output_n
        else:
          return output_p, output_n 
    
