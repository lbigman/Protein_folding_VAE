
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
        self.fc31 = nn.Linear(50, latent_dim)
        self.fc32 = nn.Linear(50, latent_dim)
       
        self.relu = nn.ReLU()
        
    def forward(self,x):
        
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        
        return self.fc31(out), self.fc32(out)
    
class Decoder(nn.Module):
#    def __init__(self,numOfContact,latent_dim):
    def __init__(self):

        super(Decoder,self).__init__()
        self.fc1 = nn.Linear(latent_dim, 50)
        self.fc2 = nn.Linear(50, 400)
        self.fc3 = nn.Linear(400, numOfContact)
        self.fc4 = nn.Linear(400, numOfContact)
        
        self.relu = nn.ReLU()
        
        
    def forward(self,x):
        
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        
        return self.fc3(out), self.fc4(out)
    

class VAE(nn.Module):
#    def __init__(self,numOfContact,latent_dim):
    def __init__(self):

        super(VAE,self).__init__()
#        super(AutoEncoder, self).__init__()
#        self.enc = Encoder(numOfContact,latent_dim)
#        self.dec = Decoder(numOfContact,latent_dim)
        self.enc = Encoder()
        self.dec = Decoder()
        
    def pick_random(self, mu, logvar):

        std = torch.exp(0.5*logvar)

        eps = torch.randn_like(std)

        return mu + eps*std

    def forward(self, x, ret_latent = False):
               
        mu, logvar = self.enc(x.view(-1, numOfContact))
        
        z = self.pick_random(mu, logvar)
        
        mu_x, logvar_x = self.dec(z)
        
        x_reco = self.pick_random(mu_x, logvar_x)

        if ret_latent:
            return latent
        
        return x_reco, mu, logvar
   
    
       
        