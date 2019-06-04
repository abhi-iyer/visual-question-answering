import os
import numpy as np
import shutil
import time
import json
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision as tv
import nntools as nt
import torch

from models import *
from preprocess import *

class SANExperiment():
    def __init__(self, output_dir, batch_size=10,
                 perform_validation_during_training=False,
                 lr=0.01, weight_decay=0.025,
                 num_epochs=10):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        images_dir = '/datasets/ee285f-public/VQA2017/'
        q_dir = '/datasets/ee285f-public/VQA2017/v2_OpenEnded_mscoco_'
        ans_dir = '/datasets/ee285f-public/VQA2017/v2_mscoco_'
        
        self.train_set = MSCOCODataset(images_dir, q_dir, 
                              ans_dir, mode='train', 
                              image_size=(448, 448))
        
        indices = np.random.permutation(len(self.train_set))
        train_ind = indices[:int(len(indices)*0.8)]
        val_ind = indices[int(len(indices)*0.8):]
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_ind)
        val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_ind)

        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=batch_size, 
                                                        pin_memory=True, 
                                                        sampler=train_sampler)
        self.val_loader = torch.utils.data.DataLoader(self.train_set, batch_size=batch_size, 
                                                      pin_memory=True, 
                                                      sampler=val_sampler)
        
        self.model = SAN(num_classes=1000, batch_size=batch_size, 
                         vocab_size=len(self.train_set.vocab_q), embedding_dim=1000,
                         output_vgg=1024, input_attention=1024, output_attention=512).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                          lr=lr, 
                                          weight_decay=weight_decay)
        
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.history = []
        self.train_loss = []
        self.train_acc = []
        
        os.makedirs(output_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(output_dir, 
                                       "checkpoint.pth.tar")
        self.config_path = os.path.join(output_dir, "config.txt")
        
        # Transfer all local arguments/variables into attributes
        locs = {k: v for k, v in locals().items() if k is not 'self'}
        self.__dict__.update(locs)
        
        if os.path.isfile(self.config_path):
            with open(self.config_path, 'r') as f:
                if f.read()[:-1] != repr(self):
                    raise ValueError(
                        "Cannot create this experiment: "
                        "I found a checkpoint conflicting with the current setting.")
            self.load()
        else:
            self.save()
    
    @property
    def epoch(self):
        return len(self.history)

    def setting(self):
        return {'Net': self.model,
                'Train Set': self.train_set,
                'Optimizer': self.optimizer,
                'BatchSize': self.batch_size}
    
    def __repr__(self):
        """Pretty printer showing the setting of the experiment. This is what
        is displayed when doing ``print(experiment)``. This is also what is
        saved in the ``config.txt`` file.
        """
        string = ''
        for key, val in self.setting().items():
            string += '{}({})\n'.format(key, val)
        return string
    
    def state_dict(self):
        """Returns the current state of the experiment."""
        return {'Net': self.model.state_dict(),
                'Optimizer': self.optimizer.state_dict(),
                'History': self.history,
                'TrainLoss' : self.train_loss,
                'TrainAcc' : self.train_acc}
    
    def load_state_dict(self, checkpoint):
        self.model.load_state_dict(checkpoint['Net'])
        self.optimizer.load_state_dict(checkpoint['Optimizer'])
        self.history = checkpoint['History']
        self.train_loss = checkpoint['TrainLoss']
        self.train_acc = checkpoint['TrainAcc']
        
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.model.device)
    
    def save(self):
        """Saves the experiment on disk, i.e, create/update the last checkpoint."""
        torch.save(self.state_dict(), self.checkpoint_path)
        with open(self.config_path, 'w') as f:
            print(self, file=f)

    def load(self):
        """Loads the experiment from the last checkpoint saved on disk."""
        checkpoint = torch.load(self.checkpoint_path,
                                map_location=self.device)
        self.load_state_dict(checkpoint)
        del checkpoint
    
    def run(self):
        self.model.train()
        
        loader = self.train_loader
        
        start_epoch = self.epoch
        print("Start/Continue training from epoch {}".format(start_epoch))
        
        for epoch in range(start_epoch, self.num_epochs):
            running_loss, running_acc, num_updates = 0.0, 0.0, 0.0
            
            for i, q, a in self.train_loader:
                if (self.device == 'cuda'):
                    i, q, a = i.cuda(), q.cuda(), a.cuda()
                
                i, q, a = Variable(i), Variable(q), Variable(a)
                
                self.optimizer.zero_grad()
                predicted_answer = self.model.forward(i, q)
                
                _, class_ind = torch.max(a, 1)
                _, y_pred = torch.max(predicted_answer, 1)
                
                loss = self.criterion(predicted_answer, class_ind)
                loss.backward()
                self.optimizer.step()
                
                with torch.no_grad():
                    running_loss += loss.item()
                    running_acc += torch.sum((y_pred == class_ind).data)
                
                num_updates += 1
                
            loss = running_loss / num_updates
            acc = (running_acc / len(loader.dataset)) * 100
            
            self.history.append(epoch)
            self.train_loss.append(loss)
            self.train_acc.append(acc)
            
            self.save()
        
        print("Finish training for {} epochs".format(self.num_epochs))

exp = SANExperiment(output_dir="exp")
exp.run()