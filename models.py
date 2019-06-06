import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import variable
import torchvision as tv
import nntools as nt
import nntools as mnt
import torch

class VGGNet(nn.Module):
    def __init__(self, output_features, fine_tuning=False):
        super(VGGNet, self).__init__()
        vgg = tv.models.vgg16_bn(pretrained=True)
        
        #freezing the feature extraction layers
        for param in vgg.parameters():
            param.requires_grad = fine_tuning
            
        self.features = vgg.features
        
        self.num_fts = 512
        self.output_features = output_features
        
        # Linear layer goes from 512 to 1024
        self.classifier = nn.Linear(self.num_fts, self.output_features)
        nn.init.xavier_uniform_(self.classifier.weight)
        
        self.tanh = nn.Tanh()
        
    def forward(self, x):        
        h = self.features(x)
                
        h = h.view(-1, 49, self.num_fts)
        
        h = self.classifier(h)
        
        y = self.tanh(h)
        
        return y
    
    
class LSTM(nn.Module): 
    def __init__(self, vocab_size, embedding_dim, num_layers=1, batch_size=100, hidden_dim=1024):
        super(LSTM,self).__init__()
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
                
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        nn.init.xavier_uniform_(self.embed.weight)
        
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, 
                            num_layers=num_layers)
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0)
        nn.init.xavier_uniform_(self.lstm.weight_hh_l0)

    
    def forward(self, q_ind, seq_length):
        embedding = self.embed(q_ind)
                
        embedding = nn.utils.rnn.pack_padded_sequence(embedding, seq_length, batch_first=True)
        
        _, h = self.lstm(embedding)
        
        return h[0][0] # return final hidden state of LSTM
    

class AttentionNet(nn.Module):
    def __init__(self, num_classes, batch_size, input_features=1024, output_features=512):
        #v_i in dxm => 1024x196 vec
        #v_q in d => 1024x1 vec
        #Wia v_i in kxm => kx196
        #will choose k => 512
        
        super(AttentionNet,self).__init__()
        self.input_features = input_features
        self.output_features = output_features #k 
        self.num_classes = num_classes
        self.batch_size = batch_size
        
        self.image1 = nn.Linear(input_features, output_features, bias=False)
        self.question1 = nn.Linear(input_features, output_features)
        self.attention1 = nn.Linear(output_features, 1)
        nn.init.xavier_uniform_(self.image1.weight)
        nn.init.xavier_uniform_(self.question1.weight)
        nn.init.xavier_uniform_(self.attention1.weight)

        
        self.image2 = nn.Linear(input_features, output_features, bias=False)
        self.question2 = nn.Linear(input_features, output_features)
        self.attention2 = nn.Linear(output_features, 1)
        nn.init.xavier_uniform_(self.image2.weight)
        nn.init.xavier_uniform_(self.question2.weight)
        nn.init.xavier_uniform_(self.attention2.weight)
        
        self.answer_dist = nn.Linear(input_features, self.num_classes)
        nn.init.xavier_uniform_(self.answer_dist.weight)
                
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, image, question):
        # image_vec = 10x196x1024
        # question_vec = 10x1024
        
        irep_1 = self.image1(image)
        qrep_1 = self.question1(question).unsqueeze(dim=1) 
        ha_1 = self.tanh(irep_1 + qrep_1)
        ha_1 = self.dropout(ha_1)
        pi_1 = self.softmax(self.attention1(ha_1))
        u_1 = (pi_1 * image).sum(dim=1) + question
        
        irep_2 = self.image2(image)
        qrep_2 = self.question2(u_1).unsqueeze(dim=1)
        ha_2 = self.tanh(irep_2 + qrep_2)
        ha_2 = self.dropout(ha_2)
        pi_2 = self.softmax(self.attention2(ha_2))
        u_2 = (pi_2 * image).sum(dim=1) + u_1
        
        w_u = self.answer_dist(self.dropout(u_2))

        return w_u

    

class SAN(nn.Module):
    def __init__(self, output_vgg, vocab_size, batch_size, embedding_dim, num_classes, 
                 input_attention, output_attention, fine_tuning=False):
        super(SAN, self).__init__()
        #output_featured -> 1024
        self.vgg = VGGNet(output_vgg)
        
        #vocab_size,embedding_dim =1000
        self.lstm = LSTM(vocab_size=vocab_size, embedding_dim=embedding_dim, 
                         batch_size=batch_size)
        
        #num_classes = 10000
        self.attention = AttentionNet(num_classes=num_classes, batch_size=batch_size, 
                                      input_features=input_attention, output_features=output_attention)
        
    def forward(self, image, question, seq_length):
        #image_embedding -> 1024x196
        image_embedding = self.vgg(image)
        
        #question_embedding -> 1024x1 
        question_embedding = self.lstm(question, seq_length)
        
        #should return answer distribution 1000x1 
        return self.attention(image_embedding, question_embedding)