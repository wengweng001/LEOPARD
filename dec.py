import os
import time
import torch
import argparse
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score, precision_score, recall_score

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


nmi = normalized_mutual_info_score
ari = adjusted_rand_score

class DEC_AE(nn.Module):
    """
    DEC auto encoder - this class is used to 
    """
    def __init__(self, nInput, nNodes, nOutput, device='cpu'):
        super(DEC_AE, self).__init__()
        # hidden layer 1
        self.linear1 = nn.Linear(nInput, nNodes, bias=True)
        nn.init.xavier_uniform_(self.linear1.weight)
        self.linear1.bias.data.zero_()

        # hidden layer 2
        self.linear2 = nn.Linear(nNodes, nOutput, bias=True)
        nn.init.xavier_uniform_(self.linear2.weight)
        self.linear2.bias.data.zero_()

        # activation function
        self.activation1 = nn.ReLU(inplace=True)
        self.activation2 = nn.Sigmoid()

        # decoder
        self.biasDecoder1 = nn.Parameter(torch.zeros(nInput))
        self.biasDecoder2 = nn.Parameter(torch.zeros(nNodes))

        self.device = device

        self.alpha = 1.0
        self.clusterCenter = nn.Parameter(torch.zeros(5,nOutput))
        self.pretrainMode = True
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)

    def setPretrain(self,mode):
        """To set training mode to pretrain or not, 
        so that it can control to run only the Encoder or Encoder+Decoder"""
        self.pretrainMode = mode
    def updateClusterCenter(self, cc):
        """
        To update the cluster center. This is a method for pre-train phase.
        When a center is being provided by kmeans, we need to update it so
        that it is available for further training
        :param cc: the cluster centers to update, size of num_classes x num_features
        """
        self.clusterCenter.data = torch.from_numpy(cc)
    def getTDistribution(self,x, clusterCenter):
        """
        student t-distribution, as same as used in t-SNE algorithm.
         q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
         
         :param x: input data, in this context it is encoder output
         :param clusterCenter: the cluster center from kmeans
         """
        xe = torch.unsqueeze(x,1).to(self.device) - clusterCenter.to(self.device)
        q = 1.0 / (1.0 + (torch.sum(torch.mul(xe,xe), 2) / self.alpha))
        q = q ** (self.alpha + 1.0) / 2.0
        q = (q.t() / torch.sum(q, 1)).t() #due to divison, we need to transpose q
        return q

    def getDistance(self,x, clusterCenter,alpha=1.0):
        """
        it should minimize the distince to 
         """
        if not hasattr(self, 'clusterCenter'):
            self.clusterCenter = nn.Parameter(torch.zeros(5,5))
        xe = torch.unsqueeze(x,1).to(self.device) - clusterCenter.to(self.device)
        # need to sum up all the point to the same center - axis 1
        d = torch.sum(torch.mul(xe,xe), 2)
        return d
        
    def forward(self,x):
        # encoder
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation1(x)  # encoded output
        x_e = x
        #if not in pretrain mode, we need encoder and t distribution output
        if self.pretrainMode == False:
            return x, self.getTDistribution(x,self.clusterCenter), self.getDistance(x_e,self.clusterCenter),F.softmax(x_e,dim=1)
        ##### encoder is done, followed by decoder #####
        # decoder
        x = F.linear(x, self.linear2.weight.t()) + self.biasDecoder2
        x = self.activation1(x)
        x = F.linear(x, self.linear1.weight.t()) + self.biasDecoder1
        x = self.activation2(x)  # reconstructed input for end-to-end
        x_de = x

        return x_e, x_de

class DEC:
    """The class for controlling the training process of DEC"""
    def __init__(self,n_clusters,n_features,alpha=1.0):
        self.n_clusters=n_clusters
        self.n_features=n_features
        self.alpha = alpha        
    @staticmethod
    def target_distribution(q):
        weight = (q ** 2  ) / q.sum(0)
        #print('q',q)
        return Variable((weight.t() / weight.sum(1)).t().data, requires_grad=True)
    @staticmethod
    def kld(q,p):
        return torch.sum(p*torch.log(p/q),dim=-1)
    @staticmethod
    def cross_entropy(q,p):
        return torch.sum(torch.sum(p*torch.log(1/(q+1e-7)),dim=-1))
    @staticmethod
    def depict_q(p):
        q1 = p / torch.sqrt(torch.sum(p,dim=0))
        qik = q1 / q1.sum()
        return qik
    @staticmethod
    def distincePerClusterCenter(dist):
        totalDist =torch.sum(torch.sum(dist, dim=0)/(torch.max(dist) * dist.size(1)))
        return totalDist

    def validateOnCompleteTestData(self, data, label, model):
        start_time = time.time()
        model.eval()

        to_eval = np.array([model(d.to(self.device))[0].data.cpu().numpy() for i,d in enumerate(data)])
        true_labels = np.array([d.cpu().numpy() for i,d in enumerate(label)])
        km = KMeans(n_clusters=len(np.unique(true_labels)), n_init=10)
        y_pred = km.fit_predict(to_eval)
        correct             = (y_pred == true_labels).sum().item()
        accuracy       = 100*correct/(y_pred == true_labels).shape[0]  # 1: correct, 0: wrong

        f1 = f1_score(true_labels, y_pred, average='weighted')
        precision = precision_score(true_labels, y_pred, average='weighted')
        recall = recall_score(true_labels, y_pred, average='weighted')
        end_time     = time.time()
        test_time = end_time - start_time

        return [accuracy, f1, precision, recall, test_time, nmi(true_labels, y_pred)]

    def pretrain(self, train_data, train_label, nExtractedFeature, epochs, batchSize, device):
        self.device = device
        self.batch_size = batchSize
        
        nData = train_data.shape[0]

        dec_ae = DEC_AE(self.n_features,nNodes=self.n_features, nOutput=nExtractedFeature, device=device) #auto encoder
        dec_ae = dec_ae.to(self.device)
        mseloss = nn.MSELoss()
        optimizer = optim.SGD(dec_ae.parameters(),lr = 0.01, momentum=0.95)
        best_acc = 0.0
        for epoch in range(epochs):
            if epoch%20 == 0:
                print('Epoch: ', epoch)
            dec_ae.train()
            running_loss=0.0
            shuffled_indices = torch.randperm(nData)
            batchs = int(nData/self.batch_size)
            for i in range(batchs):
                # load data
                indices = shuffled_indices[i*self.batch_size:((i+1) * self.batch_size)]
                x = train_data[indices]
                label = train_label[indices]
                x, label = Variable(x).to(self.device),Variable(label).to(self.device)
                optimizer.zero_grad()
                x_ae,x_de = dec_ae(x)
                loss = F.mse_loss(x_de,x,reduction='mean') 
                loss.backward()
                optimizer.step()
                x_eval = x.data.cpu().numpy()
                label_eval = label.data.cpu().numpy()
                running_loss += loss.data.cpu().numpy()
                if i % 100 == 99:    # print every 100 mini-batches
                    print('[%d, %5d] loss: %.7f' %
                          (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0
            #now we evaluate the accuracy with AE
            dec_ae.eval()
            metrics = self.validateOnCompleteTestData(train_data, train_label, dec_ae)
            currentAcc = metrics[0]
            if epoch == (epoch-1):
                print('Pre-Training accuracy on source: ', currentAcc)
            if currentAcc > best_acc:
                file_path = 'models/ae'
                if not os.path.exists(file_path):
                    os.makedirs(file_path)
                best_model = 'models/ae/bestModel{}'.format(best_acc)                
                torch.save(dec_ae, best_model)
                best_acc = currentAcc
        return best_model

    def clustering(self,mbk,x,model):
        model.eval()
        y_pred_ae,_,_,_ = model(x)
        y_pred_ae = y_pred_ae.data.cpu().numpy()
        y_pred = mbk.partial_fit(y_pred_ae) #seems we can only get a centre from batch
        self.cluster_centers = mbk.cluster_centers_ #keep the cluster centers
        model.updateClusterCenter(self.cluster_centers)
    
    def trainae(self, model, data, epochs, lr):

        model.setPretrain(True)
        model = model.to(self.device)
        nData = data.shape[0]

        optimizer = optim.SGD([\
             {'params': model.parameters()}, \
            ],lr = lr, momentum=0.9)

        for epoch in range(epochs):
            model.train()
            shuffled_indices = torch.randperm(nData)
            batchs = int(nData/self.batchsize)
            for i in range(batchs):
                # load data
                indices = shuffled_indices[i*self.batchsize:((i+1) * self.batchsize)]
                x = data[indices]

                x = Variable(x).to(self.device)
                optimizer.zero_grad()
                x_ae,x_de = model(x)

                loss = F.mse_loss(x_de,x,reduce=True) 
                loss.backward()
                optimizer.step()
                
        return model
        
    def train(self, batchDataS, source_label, batchDataT, target_label, epochs, lr, best_model):
        """This method will start training for DEC cluster"""
        ct = time.time()
        
        batchData = torch.cat((batchDataS, batchDataT), dim=0)
        shuffledList = torch.randperm(batchData.shape[0])
        batchData = batchData[shuffledList, :]  # Target and Source data are shuffled together

        nData = batchData.shape[0]

        model = torch.load(best_model).to(self.device)
        model.setPretrain(False)
        optimizer = optim.SGD([\
             {'params': model.parameters()}, \
            ],lr = lr, momentum=0.95)
        print('Initializing cluster center with pre-trained weights')
        mbk = MiniBatchKMeans(n_clusters=self.n_clusters, n_init=10, batch_size=self.batch_size)
        got_cluster_center = False
        for epoch in range(epochs):
            print('Epoch: ', epoch)

            shuffled_indices = torch.randperm(nData)

            batchs = int(nData/self.batch_size)
            for i in range(batchs):
                # load data
                indices = shuffled_indices[i*self.batch_size:((i+1) * self.batch_size)]
                x = batchData[indices]
                x = Variable(x).to(self.device)
                optimizer.zero_grad()
                #step 1 - get cluster center from batch
                #here we are using minibatch kmeans to be able to cope with larger dataset.
                if not got_cluster_center:
                    self.clustering(mbk,x,model)
                    if epoch > 1:
                        got_cluster_center = True
                else:
                    model.train()
                    #now we start training with acquired cluster center
                    feature_pred,q,dist,clssfied = model(x)
                    d = self.distincePerClusterCenter(dist)
                    qik = self.depict_q(clssfied)
                    loss1 = self.cross_entropy(clssfied,qik)
                    loss = d + loss1
                    loss.backward()
                    optimizer.step()
                    
            model.setPretrain(True)
            metrics_S = self.validateOnCompleteTestData(batchDataS, source_label,model)    
            metrics_T = self.validateOnCompleteTestData(batchDataT, target_label,model) 
        
        return model, metrics_S, metrics_T