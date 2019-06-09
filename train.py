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
import torch.utils.data as data

from models import *
from preprocess import *

images_dir = '/datasets/ee285f-public/VQA2017/'
q_dir = '/datasets/ee285f-public/VQA2017/v2_OpenEnded_mscoco_'
ans_dir = '/datasets/ee285f-public/VQA2017/v2_mscoco_'

train_set = MSCOCODataset(images_dir, q_dir, 
                          ans_dir, mode='train', 
                          image_size=(224, 224))

def collate_fn(batch):
    batch.sort(key=lambda x : x[2], reverse=True)
    return data.dataloader.default_collate(batch)

class SANExperiment():
    def __init__(self, train_set, output_dir, batch_size=200, num_epochs=10):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.train_set = train_set
        
        torch.backends.cudnn.benchmark = False
        
        self.indices = np.random.permutation(len(self.train_set))
        self.indices = self.indices[:int(len(self.indices)*0.5)]
                
        train_ind = self.indices[:int(len(self.indices)*0.8)]
        val_ind = self.indices[int(len(self.indices)*0.8):]
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_ind)
        val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_ind)

        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=batch_size, 
                                                        sampler=train_sampler,
                                                        collate_fn=collate_fn)
        self.val_loader = torch.utils.data.DataLoader(self.train_set, batch_size=batch_size, 
                                                      sampler=val_sampler,
                                                      collate_fn=collate_fn)
        
        
        self.image_model = VGGNet(output_features=1024).to(self.device)
        self.question_model = LSTM(vocab_size=len(self.train_set.vocab_q), embedding_dim=1000,
                                   batch_size=batch_size, hidden_dim=1024).to(self.device)
        self.attention = AttentionNet(num_classes=1000, batch_size=batch_size,
                                      input_features=1024, output_features=512).to(self.device)
        
        self.optimizer_parameter_group = [{'params': self.question_model.parameters()}, 
                                          {'params': self.image_model.parameters()},
                                          {'params': self.attention.parameters()}]

        self.criterion = nn.CrossEntropyLoss()
        
        self.optimizer = torch.optim.RMSprop(self.optimizer_parameter_group,
                                             lr=4e-4, alpha=0.99, eps=1e-8, momentum=0.9)
        
        self.total_ex = len(train_ind)
        
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
        return {'ImageModel': self.image_model,
                'QuestionModel' : self.question_model,
                'AttentionModel' : self.attention,
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
        return {'ImageModel': self.image_model.state_dict(),
                'QuestionModel' : self.question_model.state_dict(),
                'AttentionModel' : self.attention.state_dict(),
                'Optimizer': self.optimizer.state_dict(),
                'History': self.history,
                'TrainLoss' : self.train_loss,
                'TrainAcc' : self.train_acc,
                'Indices' : self.train_ind}
    
    def load_state_dict(self, checkpoint):
        self.image_model.load_state_dict(checkpoint['ImageModel'])
        self.question_model.load_state_dict(checkpoint['QuestionModel'])
        self.attention.load_state_dict(checkpoint['AttentionModel'])
        self.optimizer.load_state_dict(checkpoint['Optimizer'])
        self.history = checkpoint['History']
        self.train_loss = checkpoint['TrainLoss']
        self.train_acc = checkpoint['TrainAcc']
        self.train_ind = checkpoint['Indices']
        
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
    
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
        
    def evaluate(self):
        self.image_model.eval()
        self.question_model.eval()
        self.attention.eval()
        
        loader = self.val_loader
        
        loss, acc = 0.0, 0.0
        
        with torch.no_grad(): 
            for i, q, s, a in loader:
                if (self.device == 'cuda'):
                    i, q, s, a = i.cuda(), q.cuda(), s.cuda(), a.cuda()

                i, q, s, a = Variable(i), Variable(q), Variable(s), Variable(a, required_grad=False)
                
                image_embed = self.image_model(i)
                question_embed = self.question_model(q.long(), s.long())
                output = self.attention(image_embed, question_embed)
                
                _, y_pred = torch.max(output, 1)

                loss += self.criterion(output, a.long().squeeze(dim=1)).item()
                acc += torch.sum((y_pred == a.long()).data)
        
        loss = (float(loss) / float(len(self.val_ind)))
        acc = (float(acc) / float(len(self.val_ind)))
        
        print("Validation Loss:", loss)
        print("Validation Accuracy:", acc)
            
    
    def run(self):
        self.image_model.train()
        self.question_model.train()
        self.attention.train()
        
        loader = self.train_loader
                
        start_epoch = self.epoch
        print("Start/Continue training from epoch {}".format(start_epoch))
        for epoch in range(start_epoch, self.num_epochs):
            running_loss, running_acc, num_updates = 0.0, 0.0, 0.0
            
            counter = 0
            
            for i, q, s, a in loader:                
                if (self.device == 'cuda'):
                    i, q, s, a = i.cuda(), q.cuda(), s.cuda(), a.cuda()
                        
                i, q, s, a = Variable(i), Variable(q), Variable(s), Variable(a)
                                
                self.optimizer.zero_grad()
                                
                image_embed = self.image_model(i)
                question_embed = self.question_model(q.long(), s.long())
                output = self.attention(image_embed, question_embed)
                
                _, y_pred = torch.max(output, 1)
                                                
                try:
                    loss = self.criterion(output, a.long().squeeze(dim=1))
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        if hasattr(torch.cuda, 'emtpy_cache'):
                            torch.cuda.empty_cache()
                    else:
                        raise e
                        
                loss.backward()
                self.optimizer.step()
                
                with torch.no_grad():
                    running_loss += loss.item()
                    running_acc += torch.sum((y_pred == a.long()).data)
                    
                num_updates += 1
                
                print("Epoch: {}, Batch: {}, Loss = {}, Acc = {}".format(epoch, counter, 
                                                              (float(running_loss) / float(num_updates * self.batch_size)),
                                                              (float(running_acc) / float(num_updates * self.batch_size))))
                
                torch.cuda.empty_cache()
                
                if (counter % 50 == 0):
                    self.history.append(epoch)
                    self.train_loss.append(float(running_loss) / float(num_updates * self.batch_size))
                    self.train_acc.append(float(running_acc) / float(num_updates * self.batch_size))
                    self.save()
                
                counter += 1
            
            loss = (float(running_loss) / float(self.total_ex))
            acc = (float(running_acc) / float(self.total_ex))
            
            print("Done with Epoch {}. Loss={}, Acc={}".format(epoch, loss, acc))
            
            self.history.append(epoch)
            self.train_loss.append(loss)
            self.train_acc.append(acc)
            
            self.save()
        
        print("Finish training for {} epochs".format(self.num_epochs)) 
        
exp = SANExperiment(output_dir="exp_batch200", train_set=train_set)
exp.run()
