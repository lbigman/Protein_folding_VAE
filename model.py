
import torch
import numpy as np
import torch.nn as nn

latent_dim = 2
numOfContact=3136
class Encoder(nn.Module):
#    def __init__(self,numOfContact,latent_dim):
    def __init__(self):    
        super(Encoder,self).__init__()
        self.fc1 = nn.Linear(numOfContact, 400)
        self.fc2 = nn.Linear(400, 50)
        self.fc3 = nn.Linear(50, 20)
        self.fc4 = nn.Linear(20, latent_dim)
       
        self.relu = nn.ReLU()
        
    def forward(self,x):
        
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        
        return self.fc4(out)
    
class Decoder(nn.Module):
#    def __init__(self,numOfContact,latent_dim):
    def __init__(self):

        super(Decoder,self).__init__()
        self.fc1 = nn.Linear(latent_dim, 20)
        self.fc2 = nn.Linear(20,50)
        self.fc3 = nn.Linear(50, 400)
        self.fc4 = nn.Linear(400, numOfContact)
        self.relu = nn.ReLU()
        
        
    def forward(self,x):
        
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        
        return self.fc4(out)
    

class Net(nn.Module):
#    def __init__(self,numOfContact,latent_dim):
    def __init__(self):

        super(Net,self).__init__()
#        super(AutoEncoder, self).__init__()
#        self.enc = Encoder(numOfContact,latent_dim)
#        self.dec = Decoder(numOfContact,latent_dim)
        self.enc = Encoder()
        self.dec = Decoder()
              
    def forward(self,x, ret_latent = False):
        
        latent = self.enc(x)
        if ret_latent:
            return latent
        
        out = self.dec(latent)
        
        return  out 
    
       