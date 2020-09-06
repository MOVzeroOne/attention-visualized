import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt 
from tqdm import tqdm
from collections import deque
import numpy as np
import matplotlib.cm as cm
from itertools import chain

class attention_mechanism(nn.Module):
    def __init__(self):
        super().__init__()
        self.a_key = nn.Parameter(torch.rand(1,2))
        self.b_key = nn.Parameter(torch.rand(1,2))
        self.a_value = torch.tensor([[1.0]])
        self.b_value = torch.tensor([[-1.0]])
        
        self.d = torch.tensor([len(self.get_keys())],dtype=torch.float)

    def forward(self,query):
        attention_vec = F.softmax(torch.matmul(query,self.get_keys())/torch.sqrt(self.d),dim=1) 
        
        return (torch.matmul(attention_vec,self.get_values()),attention_vec)
    
    def get_values(self):
        return torch.cat([self.a_value,self.b_value])

    def get_keys(self):
        return torch.cat([self.a_key,self.b_key]).transpose(0,1)

class embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Parameter(torch.rand(1,2))

class visualizer():
    def __init__(self,embed,att_mech):
        self.fig, self.axis = plt.subplots(3)

        self.embed = embed
        self.att_mech = att_mech
        
        self.episode = deque(maxlen=100)
        self.output = deque(maxlen=100)
        self.target = deque(maxlen=100)

    def arrows(self):
        vec_a = self.att_mech.a_key.view(-1).detach().numpy()
        vec_b = self.att_mech.b_key.view(-1).detach().numpy()
        vec_query = self.embed.query.view(-1).detach().numpy()

        self.axis[1].quiver(0,0,vec_a[0],vec_a[1],color='blue',label=r"$\vec{a}$")
        self.axis[1].quiver(0,0,vec_b[0],vec_b[1],color='red',label=r"$\vec{b}$")
        self.axis[1].quiver(0,0,vec_query[0],vec_query[1],color='green',label=r"$\vec{query}$")
        self.axis[1].legend()
    
    
    def bar(self,attention):
        self.axis[0].bar(("a","b"),attention.detach().numpy().flatten())
        
    
    def output_target(self):
        self.axis[2].plot(self.episode,self.output,label="output")
        self.axis[2].plot(self.episode,self.target,label="target")
        self.axis[2].legend()
    
    def append(self,episode,target,output):
        self.target.append(target.detach().numpy().flatten().item())
        self.output.append(output.detach().numpy().flatten().item())
        self.episode.append(episode)

    def cla(self):
        for ax in self.axis.flatten():
            ax.cla()

if __name__ == "__main__":
    att_mech = attention_mechanism()
    embed = embedding()
    target = torch.tensor([[1.0]])

    
    EPOCHS = 1000
    optimizer = optim.Adam(chain(*[att_mech.parameters(),embed.parameters()]),lr=0.1)
    

    
    plt.ion()
    vis = visualizer(embed,att_mech)

    for i in range(EPOCHS):
        optimizer.zero_grad()

        output,attention_vec = att_mech(embed.query)

        loss = nn.MSELoss()(output,target)
        loss.backward()
        optimizer.step()

        #visualization
        vis.append(i,target,output)
        vis.output_target()
        vis.arrows()
        vis.bar(attention_vec)
        plt.pause(0.1)
        vis.cla()