from torch import nn
from torch.nn import functional as F
import os
import sys
current_dir = ""  #path of your current directory
layer_dir = os.path.join(current_dir, 'PI2_Layers')
sys.path.append(layer_dir)
print(layer_dir)
from PI2_FC_in import  MPLayer_in_K
from PI2_CONV_in import temp_conv_k_opt

# LeNet-5 conventional Model 
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.dropout = nn.Dropout(0.05)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# LeNet-5 PI^2 Model
class LeNet5_K(nn.Module):
    def __init__(self,gamma=0,temp1=False,temp2=False,temp3=False,temp4=False,temp5=False):
        super(LeNet5_K, self).__init__()
        self.temp1 = temp1
        self.temp2 = temp2
        self.temp3 = temp3
        self.temp4 = temp4
        self.temp5 = temp5
        if(temp1):
            print("temp1")
            self.conv1 = temp_conv_k_opt(in_channels=1,out_channels=6,kernel_size=5,gamma=gamma[0],padding=0)
        else:
            self.conv1 = nn.Conv2d(1, 6, kernel_size=5,bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        if(temp2):
            print("temp2")
            self.conv2 = temp_conv_k_opt(in_channels=6,out_channels=16,kernel_size=5,gamma=gamma[1],padding=0)
        else:
            self.conv2 = nn.Conv2d(6, 16, kernel_size=5,bias=False)
        
        if(temp3):
            print("temp3")
            self.fc1 =  MPLayer_in_K(inp_node=16 * 4 * 4,out_node= 120,gamma=gamma[2])
        else:
            self.fc1 = nn.Linear(16 * 4 * 4, 120,bias=False)
        
        if(temp4):
            print("temp4")
            self.fc2 =  MPLayer_in_K(inp_node=120,out_node= 84,gamma=gamma[3])
        else:
            self.fc2 = nn.Linear(120, 84,bias=False)
        self.dropout = nn.Dropout(0.2)
        
        if(temp5):
            print("temp5")
            self.fc3 =  MPLayer_in_K(inp_node=84,out_node=10,gamma=gamma[4])
        else:
            self.fc3 = nn.Linear(84, 10,bias=False)

    def forward(self, x):
        if(self.temp1):
            x = F.relu(-1*self.conv1(x)) #25*1
        else:
            x = F.relu(self.conv1(x)) #25*1
        x = self.pool(x)
        
        if(self.temp2):
            x = F.relu(-10*self.conv2(x)) #25*1
        else:
            x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = x.reshape(x.size(0), -1)
        
        if(self.temp3):
            x = F.relu(-10*self.fc1(x))
        else:
            x = F.relu(self.fc1(x))
            
        if(self.temp4):
            x = F.relu(-10*self.fc2(x))
        else:
            x = F.relu(self.fc2(x))
        x = self.dropout(x)
        if(self.temp5):
            x = F.relu(-10*self.fc3(x))
        else:
            x = self.fc3(x)
        return x
