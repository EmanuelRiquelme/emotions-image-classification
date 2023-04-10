import torch
import torch.nn as nn
from dataset import Emotions
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import trange
from cnn_model import Model as model
import os
from utils import validation,save_model,load_model

torch.multiprocessing.set_sharing_strategy('file_system')
train_set = Emotions('train') 
val_set = Emotions('test') 

batch_size = 64
train_set = DataLoader(train_set,batch_size = batch_size,num_workers = 4,shuffle = True)
val_set = DataLoader(val_set,batch_size = batch_size, num_workers = 4,shuffle = True)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = model().to(device)
opt = optim.Adam(model.parameters(),lr=0.0001)
loss_fn = nn.CrossEntropyLoss()

epochs = 30

def train(train_set = train_set,model = model,loss_fn = loss_fn,
                opt = opt,epochs = epochs,val_set = val_set,threshold = .75):
    for epoch in  (t := trange(epochs)):
        it = iter(train_set)
        temp_loss = []
        for _ in range(len(train_set)):
            input,target = next(it)
            input,target = input.to(device),target.to(device)
            opt.zero_grad()
            output = model(input)
            loss = loss_fn(output,target)
            temp_loss.append(loss.item())
            loss.backward()
            opt.step()
        val =validation(val_set,model)
        temp_loss = sum(temp_loss)/len(temp_loss)
        t.set_description(f'validation: {val:.2f},loss : {temp_loss:.2f}')
        save_model(model,opt)
        if val >= threshold:
            break

if __name__ == '__main__':
    #load_model(model,opt)
    print(f'initial val: {validation(val_set,model):.2f}')
    train(epochs = 1)
    #save_model(model,opt)
