from torch import nn
from torch.nn import functional as F
import sys
import os
import torch

current_dir = ""  #path of the current directory
layer_dir = os.path.join(current_dir, 'PI2_Layers')
sys.path.append(layer_dir)
print(layer_dir)
from PI2_CONV_in import temp_conv_k_opt
from PI2_FC_in import MPLayer_in_K


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,bias=False), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

def conv_block_Kr2(out_channels, pool=False):
    layers = [nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9_100(nn.Module):
    def __init__(self, in_channels, out=100,gamma=0,temp0 = False, temp1=False,temp2=False,temp3=False,temp4=False,temp5=False,temp6=False,temp7=False,temp8=False):
        super().__init__()
        self.temp0 = temp0
        self.temp1 = temp1
        self.temp2 = temp2
        self.temp3 = temp3
        self.temp4 = temp4
        self.temp5 = temp5
        self.temp6 = temp6
        self.temp7 = temp7
        self.temp8 = temp8
        
        if(temp0):
            self.c1 = temp_conv_k_opt(in_channels=in_channels, out_channels= 16*4, gamma=gamma[0],diff=1)
            self.conv1 = conv_block_Kr2(16*4)
        else:
            print("1")
            self.conv1 = conv_block(in_channels, 16*4)
            
        if(temp1):
            self.c2 = temp_conv_k_opt(16*4, 32*4, kernel_size=3,gamma=gamma[1],diff=1)#64*9
            self.conv2 = conv_block_Kr2(32*4, pool=True)
        else:
            print("2")
            self.conv2 = conv_block(16*4, 32*4, pool=True)
            
        if(temp2):
            self.res11 = temp_conv_k_opt(32*4, 32*4,kernel_size=3,gamma=gamma[2],diff=1)
            self.res12 = conv_block_Kr2(32*4)
        else:
            print("3")
            self.res11 = (conv_block(32*4, 32*4))
        
        if(temp3):
            self.res21 = temp_conv_k_opt(32*4, 32*4,kernel_size=3,gamma=gamma[3],diff=1)
            self.res22 = conv_block_Kr2(32*4)
        else:
            print("4")
            self.res22 = conv_block(32*4, 32*4)
        
        if(temp4):    
            self.c3 = temp_conv_k_opt(32*4, 64*4, kernel_size=3,gamma=gamma[4],diff=1)#64*9
            self.conv3 = conv_block_Kr2(64*4, pool=True)
        else:
            print("5")
            self.conv3 = conv_block(32*4, 64*4, pool=True)
        
        if(temp5):
            self.c4 = temp_conv_k_opt(64*4, 128*4, kernel_size=3,gamma=gamma[5],diff=1)#64*9
            self.conv4 = conv_block_Kr2(128*4, pool=True)
        else:
            print("6")
            self.conv4 = conv_block(64*4, 128*4, pool=True)
        
        if(temp6):
            self.res211 = temp_conv_k_opt(128*4, 128*4,kernel_size=3,gamma=gamma[6],diff=1)
            self.res212 = conv_block_Kr2(128*4)
        else:
            print("7")
            self.res211 = (conv_block(128*4, 128*4))
        
        if(temp7):
            self.res221 = temp_conv_k_opt(128*4, 128*4,kernel_size=3,gamma=gamma[7],diff=1)
            self.res222 = conv_block_Kr2(128*4)
        else:
            print("8")
            self.res222 = conv_block(128*4, 128*4)
      
        if(temp8):
            self.bn = nn.BatchNorm1d(out)
            self.classifier = nn.Sequential(nn.MaxPool2d(4), 
                                        nn.Flatten(), 
                                        MPLayer_in_K(128*4, out,gamma = gamma[8],diff=1))
        else:
            print("9")
            self.classifier = nn.Sequential(nn.MaxPool2d(4), 
                                        nn.Flatten(), 
                                        nn.Linear(128*4, out,bias=False))

        self.d = nn.Dropout2d(0.1)
        
    def forward(self, xb):
        
        if(self.temp0):
            outp,outn = self.c1(xb)
            out = self.conv1(outp - outn)
        else:
            out = self.conv1(xb)

        if(self.temp1):
            outp,outn = self.c2(out) #64*9 = 576
            out = self.conv2(outp -outn)
        else:
            out = self.conv2(out)
            
        if(self.temp2):
            outp,outn = self.res11(out)
            out1 = self.res12(outp-outn)
        else:
            out1 = self.res11(out)
        
        if(self.temp3):
            outp,outn = self.res21(out1)
            out = self.res22(outp-outn) + out
        else:
            out = self.res22(out1) + out
                
        # out = self.d(out)
        if(self.temp4):
            outp,outn = self.c3(out) #64*9 = 576
            out = self.conv3(outp - outn)
        else:
            out = self.conv3(out)
        
        if(self.temp5):
            outp,outn = self.c4(out) #64*9 = 576
            out = self.conv4(outp-outn)
        else:
            out = self.conv4(out)
            
        if(self.temp6):
            outp,outn = self.res211(out)
            out1 = self.res212(outp-outn)
        else:
            out1 = self.res211(out)
        
        if(self.temp7):
            outp,outn = self.res221(out1)
            out = self.res222(outp-outn) + out
        else:
            out = self.res222(out1) + out
        # out = self.d(out)
        if(self.temp8):
            outp,outn = self.classifier(out)
            out = outp - outn
            out = self.bn(out)
        else:
           out = self.classifier(out) 
        return out   

#pi^K model with noise, pkt_drop
class ResNet9_100_temp_noise(nn.Module):
    def __init__(self, in_channels, out=100,gamma=0,std = 0,temp0 = False, temp1=False,temp2=False,temp3=False,temp4=False,temp5=False,temp6=False,temp7=False,temp8=False,sparse=False,drop=1):
        super().__init__()
        self.temp0 = temp0
        self.temp1 = temp1
        self.temp2 = temp2
        self.temp3 = temp3
        self.temp4 = temp4
        self.temp5 = temp5
        self.temp6 = temp6
        self.temp7 = temp7
        self.temp8 = temp8
        self.std = std

        self.c1 = temp_conv_k_opt(in_channels=in_channels, out_channels= 16*4, gamma=gamma[0],diff=0,sparse=sparse,drop_prob=drop)
        self.conv1 = conv_block_Kr2(16*4)
        self.c2 = temp_conv_k_opt(16*4, 32*4, kernel_size=3,gamma=gamma[1],diff=0,sparse=sparse,drop_prob=drop)#64*9
        self.conv2 = conv_block_Kr2(32*4, pool=True)
        self.res11 = temp_conv_k_opt(32*4, 32*4,kernel_size=3,gamma=gamma[2],diff=0,sparse=sparse,drop_prob=drop)
        self.res12 = conv_block_Kr2(32*4)
        self.res21 = temp_conv_k_opt(32*4, 32*4,kernel_size=3,gamma=gamma[3],diff=0,sparse=sparse,drop_prob=drop)
        self.res22 = conv_block_Kr2(32*4)
        self.c3 = temp_conv_k_opt(32*4, 64*4, kernel_size=3,gamma=gamma[4],diff=0,sparse=sparse,drop_prob=drop)
        self.conv3 = conv_block_Kr2(64*4, pool=True)
        self.c4 = temp_conv_k_opt(64*4, 128*4, kernel_size=3,gamma=gamma[5],diff=0,sparse=sparse,drop_prob=drop)
        self.conv4 = conv_block_Kr2(128*4, pool=True)
        self.res211 = temp_conv_k_opt(128*4, 128*4,kernel_size=3,gamma=gamma[6],diff=0,sparse=sparse,drop_prob=drop)
        self.res212 = conv_block_Kr2(128*4)
        self.res221 = temp_conv_k_opt(128*4, 128*4,kernel_size=3,gamma=gamma[7],diff=0,sparse=sparse,drop_prob=drop)
        self.res222 = conv_block_Kr2(128*4)
        self.classifier = nn.Sequential(nn.MaxPool2d(4), 
                                        nn.Flatten(), 
                                        MPLayer_in_K(128*4, out,gamma = gamma[8],diff=0,sparse=sparse,drop_prob=drop))

        
    def forward(self, xb):
            out = -1*self.c1(xb)
            noise1 = torch.randn_like(out) * self.std
            out = out + noise1
            out = self.conv1(out)
            
            out= -1*self.c2(out) #64*9 = 576
            noise2 = torch.randn_like(out) * self.std
            out = out + noise2
            out = self.conv2(out)
      
            out1 = -1*self.res11(out)
            noise1 = torch.randn_like(out) * self.std
            out = out + noise1
            out1 = self.res12(out1)

            out1 = -1*self.res21(out1)
            noise1 = torch.randn_like(out) * self.std
            out = out + noise1
            out = self.res22(out1) + out
            
            out = -1*self.c3(out) #64*9 = 576
            noise1 = torch.randn_like(out) * self.std
            out = out + noise1
            out = self.conv3(out)

        
            out = -1*self.c4(out) #64*9 = 576
            noise1 = torch.randn_like(out) * self.std
            out = out + noise1
            out = self.conv4(out)

            out1 = -1*self.res211(out)
            noise1 = torch.randn_like(out) * self.std
            out = out + noise1
            out1 = self.res212(out1)

            out1 = -1*self.res221(out1)
            noise1 = torch.randn_like(out) * self.std
            out = out + noise1
            out = self.res222(out1) + out
            
            out = -10*self.classifier(out)
            noise1 = torch.randn_like(out) * self.std
            out = out + noise1
            return out   
