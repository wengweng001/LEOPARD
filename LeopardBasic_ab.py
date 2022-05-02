# The code in this file was originally developed by Andri Ashfahani with any major modification

import numpy as np
from scipy.special import softmax
import time
import copy
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import warnings
from utilsADCN import clusteringLoss, maskingNoise, imageNoise, meanStdCalculator, stableSoftmax, getlistchunk
from model import ConvAeMNIST, smallAE, cluster, ADCNoldtask, domain_classifier

from mmd import mmd_g, MMD, MMDu, MatConvert, MMDu_multi

warnings.filterwarnings("ignore", category=RuntimeWarning)


class ADCN():
    def __init__(self, nOutput, nInitCluster, nInput = 196, nHiddenNode = 96, alpha_w = 0.005, alpha_d = 0.001, LR = 0.1, desiredLabels = [0,1,2,3,4,5,6,7,8,9], alpha_kl=0.1, alpha_dc=0.1):
        
        # initial network
        self.ADCNcnn     = ConvAeMNIST()
        self.ADCNae      = [smallAE(nInput, nHiddenNode)]
        self.discriminator = domain_classifier(nInput, nHiddenNode)
        self.ADCNold     = []
        self.ADCNcluster = []

        # network significance
        self.averageBias  = [meanStdCalculator()]
        self.averageVar   = [meanStdCalculator()]
        self.averageInput = [meanStdCalculator()]

        # hyperparameters
        self.lr        = LR
        self.criterion = nn.MSELoss()

        # drift detection parameters
        self.alphaWarning      = alpha_w
        self.alphaDrift        = alpha_d
        self.driftStatusOld    = 0
        self.driftStatus       = 0
        self.driftHistory      = []
        # self.prevFeatureMatrix = []
        self.bufferData  = torch.Tensor().float()
        self.bufferLabel = torch.Tensor().long()

        # Evolving
        self.growNode       = False
        self.pruneNode      = False
        self.evolving       = True
        self.clusterGrowing = True

        # net properties
        self.nInput       = nInput
        self.nOutput      = nOutput
        self.nHiddenLayer = 1
        self.nHiddenNode  = nHiddenNode

        # cluster properties
        self.desiredLabels= desiredLabels
        self.nInitCluster = nInitCluster
        self.nCluster     = self.nInitCluster
        self.regStrClusteringLoss = 0.01

        # LWF
        self.nOutputPerTask = nOutput
        self.regStrLWF      = 5

        # params
        self.clutering_loss  = True
        self.domain_loss     = True
        self.alpha_kl        = alpha_kl
        self.alpha_dc        = alpha_dc
        if self.alpha_kl == 0.0:
            self.clutering_loss = False
        if self.alpha_dc == 0.0:
            self.domain_loss = False

    def updateNetProperties(self):
        self.nHiddenLayer = len(self.ADCNae)
        nHiddenNode = 0
        nCluster    = 0
        for idx, nett in enumerate(self.ADCNae):
            nHiddenNode  += nett.nNodes
            try:
                nCluster   += self.ADCNcluster[idx].nCluster
            except:
                nCluster    = 0

        self.nHiddenNode  = nHiddenNode
        self.nCluster     = nCluster

    def getNetProperties(self):
        for _,nett in enumerate(self.ADCNae):
            nett.getNetProperties()

    @staticmethod
    def target_distribution(x):
        weight = (x ** 2) / torch.sum(x, 0)
        return (weight.t() / torch.sum(weight,1)).t()

    # ============================= forward pass =============================
    def forwardADCN(self, x, winIdx = None):
        # prepare model
        self.ADCNcnn = self.ADCNcnn.to(self.device)

        # forward encoder CNN
        x = x.to(self.device)
        x = self.ADCNcnn(x)

        # feedforward from input layer to latent space
        for iLayer in range(len(self.ADCNae)):
            currnet = self.ADCNae[iLayer].network
            obj     = currnet.train()
            obj     = obj.to(self.device)
            x       = obj(x)

            if winIdx is not None:
                # store latent features of the winning layer for clustering
                if iLayer == winIdx:
                    self.latentFeatures = x.detach().clone()

                if iLayer == len(self.ADCNae) - 2:
                    # store input feature for the last hidden layer
                    self.lastInputFeature = x.detach().clone()

        return x

    def forwardBiasVar(self, target, winIdx):
        # x is the input features for this layer
        # only for ADCNae
        with torch.no_grad():
            target = target.to(self.device)

            # encode
            x  = self.ADCNae[winIdx].network(target)
            hiddenNodeSignificance = torch.mean(x.detach().clone(), 0)
            x2 = (x.detach().clone())**2

            # decode
            x  = self.ADCNae[winIdx].network(x, 2)
            x2 = self.ADCNae[winIdx].network(x2, 2)

            # get bias and variance
            bias     = torch.mean((x - target)**2).item()
            variance = torch.mean(x2 - target**2).item()
            # pdb.set_trace()

        return bias, variance, hiddenNodeSignificance

    def forward(self, x):
        # encode decode in end-to-end manner
        # prepare model
        self.ADCNcnn = self.ADCNcnn.to(self.device)

        # prepare data
        x = x.to(self.device)

        # forward encoder CNN
        x = self.ADCNcnn(x)

        # feedforward from input layer to latent space, encode
        for iLayer,_ in enumerate(self.ADCNae):
            currnet = self.ADCNae[iLayer].network
            obj     = currnet.train()
            obj     = obj.to(self.device)
            x       = obj(x)

            # should consider clustering loss


        # feedforward from latent space to output layer, decode
        for iLayer in range(len(self.ADCNae)-1,0-1,-1):
            currnet = self.ADCNae[iLayer].network
            obj     = currnet.train()
            obj     = obj.to(self.device)
            x       = obj(x, 2)

        # forward decoder CNN
        x = self.ADCNcnn(x, 2)

        return x

    def optimizeCNN(self, x):
        # encode decode in end-to-end manner
        # prepare model

        # prepare data
        x = x.to(self.device)

        shuffled_indices = torch.randperm(np.shape(x)[0])
        img_x = x[shuffled_indices]
        x = x[shuffled_indices]

        # get optimizer
        optimizer = torch.optim.SGD(self.ADCNcnn.parameters(), lr=self.lr, momentum=0.95, weight_decay=0.00005)
        for iLayer, _ in enumerate(self.ADCNae):
            optimizer.add_param_group({'params': self.ADCNae[iLayer].network.parameters()})

        # prepare network
        self.ADCNcnn = self.ADCNcnn.train()
        self.ADCNcnn = self.ADCNcnn.to(self.device)

        # forward encoder CNN
        x = self.ADCNcnn(x)

        # feedforward from input layer to latent space, encode
        for iLayer,_ in enumerate(self.ADCNae):
            currnet = self.ADCNae[iLayer].network
            obj     = currnet.train()
            obj     = obj.to(self.device)
            x       = obj(x)

            # should consider clustering loss


        # feedforward from latent space to output layer, decode
        for iLayer in range(len(self.ADCNae)-1,0-1,-1):
            currnet = self.ADCNae[iLayer].network
            obj     = currnet.train()
            obj     = obj.to(self.device)
            x       = obj(x, 2)

        # forward decoder CNN
        recons_x = self.ADCNcnn(x, 2)

        loss = self.criterion(recons_x, img_x)

        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # perform a single optimization step (parameter update)
        optimizer.step()


        return x

    def optimizeSource(self, x):
        # encode decode in end-to-end manner
        # prepare model
        Tensor = torch.FloatTensor
        shuffled_indices = torch.randperm(np.shape(x)[0])
        img_x = x[shuffled_indices]
        x = x[shuffled_indices]
        self.ADCNcnn = self.ADCNcnn.to(self.device)

        # get optimizer
        optimizer = torch.optim.SGD(self.ADCNcnn.parameters(), lr=self.lr, momentum=0.95, weight_decay=0.00005)
        for iLayer, _ in enumerate(self.ADCNae):
            optimizer.add_param_group({'params': self.ADCNae[iLayer].network.parameters()})

        # prepare network
        self.ADCNcnn = self.ADCNcnn.train()
        self.ADCNcnn = self.ADCNcnn.to(self.device)

        # prepare data
        x = x.to(self.device)

        # forward encoder CNN
        x = self.ADCNcnn(x)
        feat_s=x

        # feedforward from input layer to latent space, encode
        for iLayer,_ in enumerate(self.ADCNae):
            currnet = self.ADCNae[iLayer].network
            obj     = currnet.train()
            obj     = obj.to(self.device)
            x       = obj(x)
            latentAE_s = x

            # should consider clustering loss


        # feedforward from latent space to output layer, decode
        for iLayer in range(len(self.ADCNae)-1,0-1,-1):
            currnet = self.ADCNae[iLayer].network
            obj     = currnet.train()
            obj     = obj.to(self.device)
            x       = obj(x, 2)
            recons_feat_s = x

        # forward decoder CNN
        recons_x = self.ADCNcnn(x, 2)

        #recons loss cnn
        loss = self.criterion(recons_x, img_x)

        #recons loss AE
        loss.add_(self.criterion(recons_feat_s, feat_s))

        # #loss KL to align the prior to gaussian prior
        # Alpha = 0.02
        # # gaussian prior source
        # # zn_s = Variable(Tensor(np.random.normal(0, 1, (np.shape(latentAE_s)[0], np.shape(latentAE_s)[1]))))
        # # zn_t = torch.normal(0, 1, size=(np.shape(latentAE_t)[0], np.shape(latentAE_t)[1]))
        # zn_s = torch.normal(0, 1, size=(np.shape(latentAE_s)[0], np.shape(latentAE_s)[1]))
        #
        #
        # # loss.add_(Alpha,F.kl_div(latentAE_s,zn_s))
        # loss.add_(Alpha, F.kl_div(torch.log_softmax(latentAE_s, dim=1),torch.softmax(zn_s,dim=1),reduction='none').mean()) #w
        # # loss.add_(Alpha, F.kl_div(torch.log_softmax(latentAE_s, dim=1),torch.softmax(zn_s,dim=1),reduction='batchmean')) #w
        # # softmax is applied across features

        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # perform a single optimization step (parameter update)
        optimizer.step()

        return x

    def optimizeTarget(self, x):
        # encode decode in end-to-end manner
        # prepare model
        Tensor = torch.FloatTensor
        shuffled_indices = torch.randperm(np.shape(x)[0])
        img_x = x[shuffled_indices]
        x = x[shuffled_indices]
        self.ADCNcnn = self.ADCNcnn.to(self.device)
        criterionDAL = nn.L1Loss().cpu()

        # get optimizer
        optimizer = torch.optim.SGD(self.ADCNcnn.parameters(), lr=self.lr, momentum=0.95, weight_decay=0.00005)
        for iLayer, _ in enumerate(self.ADCNae):
            optimizer.add_param_group({'params': self.ADCNae[iLayer].network.parameters()})

        # prepare network
        self.ADCNcnn = self.ADCNcnn.train()
        self.ADCNcnn = self.ADCNcnn.to(self.device)

        # prepare data
        x = x.to(self.device)

        # forward encoder CNN
        x = self.ADCNcnn(x)
        feat_t=x

        # feedforward from input layer to latent space, encode
        for iLayer,_ in enumerate(self.ADCNae):
            currnet = self.ADCNae[iLayer].network
            obj     = currnet.train()
            obj     = obj.to(self.device)
            x       = obj(x)
            latentAE_t = x

            # should consider clustering loss


        # feedforward from latent space to output layer, decode
        for iLayer in range(len(self.ADCNae)-1,0-1,-1):
            currnet = self.ADCNae[iLayer].network
            obj     = currnet.train()
            obj     = obj.to(self.device)
            x       = obj(x, 2)
            recons_feat_t = x

        # forward decoder CNN
        recons_x = self.ADCNcnn(x, 2)

        #recons loss cnn
        loss = self.criterion(recons_x, img_x)

        #recons loss AE
        loss.add_(self.criterion(recons_feat_t, feat_t))

        # #loss KL to align the prior to gaussian prior
        # Alpha = 0.02
        # # gaussian prior source
        # # zn_t = Variable(Tensor(np.random.normal(0, 1, (np.shape(latentAE_t)[0], np.shape(latentAE_t)[1]))))
        # zn_t = torch.normal(0, 1, size=(np.shape(latentAE_t)[0], np.shape(latentAE_t)[1]))
        # loss.add_(Alpha,criterionDAL(latentAE_t,zn_t))

        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # perform a single optimization step (parameter update)
        optimizer.step()

        return x


    # ============================= Initialization =============================
    def initialization(self,dataStreamSource, winIdx, batchSize = 16, epoch = 50, device = torch.device('cpu'), init=False):

        # initialization phase: train CNN, train AE, train cluster, without clustering loss, evolving, epoch, add cluster in the end
        # always trained using labeled data
        # will create cluster for the last layer

        '''
        labeledDataSrc = dataStreamSource.labeledData;
        labeledDataTrg = dataStreamTarget.unlabeledData;
        '''
        labeledDataSrc=dataStreamSource.labeledData;
        #labeledDataTrg = dataStreamTarget.unlabeledData[0:200];

        labeledData=labeledDataSrc
        #labeledData = torch.cat((dataStreams.labeledData, dataStreamstwo.labeledData),dim=0)  # ini untuk sementara


        if winIdx == 0:
            self.labeledDataSrc  = labeledDataSrc
            #self.labeledDataTrg = labeledDataTrg
            self.unlabeledDataSrc = dataStreamSource.unlabeledData
            # self.unlabeledDataTrg = dataStreamTarget.unlabeledData
            #MNIST 70000

            self.batchSize    = batchSize

            self.device = device
            print('Network initialization phase is started')

            # Train CNN, for the first training, epoch is used
            extractedFeatures = self.trainCNN(labeledData, unlabeled = False, epoch = epoch)#initialization 50 epoch, data sudah gabungan target and source

            # Train AE, this trains the newly created layer
            latentFeatures = self.trainBasicAe(extractedFeatures, labeledData, evolving = self.evolving, winIdx = 0, epoch = epoch, unlabeled = False)
            # latentFeaturesSrc = self.trainBasicAeKL( dataStreamTarget.unlabeledData,  dataStreamTarget.unlabeledData, evolving = self.evolving, winIdx = 0, epoch = 1, unlabeled = True)
            # latentFeaturesTrg = self.trainBasicAeL1(extractedFeatures, self.unlabeledDataTrg, evolving=False, winIdx=0,
            #                                        epoch=epoch, unlabeled=False)
            # latentFeatures=torch.cat((latentFeaturesAE,latentFeaturesSrc,latentFeaturesTrg),dim=0)

            initialData    = latentFeatures.detach().clone().to('cpu')

            # create cluster, this is only done when there is no cluster for idx-th layer
            # self.createCluster(initialData, epoch = epoch)
            self.createCluster(initialData, epoch=epoch, initialization=init)

        if winIdx > 0:
            self.forwardADCN(labeledData, winIdx = winIdx)

            # Train AE, this trains the newly created layer
            latentFeatures = self.lastInputFeature          # latentFeatures is the extracted features from 2nd last layers
            latentFeatures = self.trainBasicAe(latentFeatures, labeledData, evolving = self.evolving, winIdx = winIdx, epoch = epoch, unlabeled = False)
            initialData    = latentFeatures.detach().clone().to('cpu')

            # create cluster, this is only done when there is no cluster for idx-th layer
            # pdb.set_trace()
            self.createCluster(initialData, epoch = epoch, initialization=init)

    def createCluster(self, initialData, epoch = 1, initialization=False):
        print('\n')
        print('Cluster initialization phase is started')

        nInput    = initialData.shape[1]

        myCluster = cluster(nInput, initialData[0:len(self.desiredLabels)],
                            nInitCluster = self.nInitCluster, clusterGrowing = self.clusterGrowing, desiredLabels = self.desiredLabels)

        # updateCluster
        myCluster.fit(initialData, epoch = epoch, init=initialization)
        self.clusterHistory = myCluster.clusterHistory
        print('A cluster was created containing ',myCluster.nCluster ,' centroids')

        # add cluster to the global cluster
        self.ADCNcluster = self.ADCNcluster + [myCluster]

    # ============================= Training =============================
    def trainCNN(self, x, unlabeled = True, epoch = 1):
        # x is image data with size 1 x 28 x 28
        nData = x.shape[0]
        x     = x.to(self.device)#x ini kalo dari initialization sudah gabungan target sama source

        # get optimizer
        optimizer = torch.optim.SGD(self.ADCNcnn.parameters(), lr = self.lr, momentum = 0.95, weight_decay = 0.00005)

        # prepare network
        self.ADCNcnn = self.ADCNcnn.train()
        self.ADCNcnn = self.ADCNcnn.to(self.device)

        # print('CNN training is started')

        for iEpoch in range(0, epoch): # ini diepoch 50 kali, kalo initialization

            # if iEpoch%20 == 0:
            #     print('Epoch: ', iEpoch)

            
            shuffled_indices = torch.randperm(nData)
            batchs = int(nData/self.batchSize)

            for i in range(batchs):

                # print('Data batch:',int(iData/self.batchSize))
                indices = shuffled_indices[i*self.batchSize:((i+1) * self.batchSize)]

                # load data
                minibatch_xTrain = x[indices]

                # clear the gradients of all optimized variables
                optimizer.zero_grad()

                # forward
                # latentAE here is the output of the deepest autoencoder
                latentFeatures = self.ADCNcnn(minibatch_xTrain)     # encode
                outputs        = self.ADCNcnn(latentFeatures, 2)    # decode

                # calculate the loss
                loss = self.criterion(outputs, minibatch_xTrain)

                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()

                # perform a single optimization step (parameter update)
                optimizer.step()

        with torch.no_grad():
            latentFeatures = self.ADCNcnn(x)

        # latentFeatures is the extracted features for the next hidden layer
        return latentFeatures.detach().clone()

    def trainBasicAe(self, x, x_ori, winIdx, epoch = 1, clustering = False, evolving = True, unlabeled = True):
        # grow layer: train CNN, train AE, train cluster
        # x is the extracted features from the previous hidden layer
        # this will update [winIdx]-th layer in greedy layer wise manner
        # TO DO: create grow and prune nodes, why batch size 1 is very bad, drift detection
        nData = x.shape[0]

        # masked input
        # maskedX = maskingNoise(x.detach().clone())              # make zero some of the input feature

        # prepare data
        x = x.to(self.device)
        x_ori = x_ori.to(self.device)
        # maskedX = maskedX.to(self.device)

        # prepare network
        self.ADCNae[winIdx].network = self.ADCNae[winIdx].network.train()
        self.ADCNae[winIdx].network = self.ADCNae[winIdx].network.to(self.device)

        self.growNode   = False
        self.pruneNode  = False
        evolve          = 0
        hiddenNodeHist  = []

        # print('AE training is started')
        for iEpoch in range(0, epoch):

            if iEpoch>0:
                evolving=False
            # print('=======Epoch==========TrainBasicAE:', iEpoch+1)

            shuffled_indices = torch.randperm(nData)
            batchs = int(nData/self.batchSize)

            for i in range(batchs):

                # print('Data batch:',int(iData/self.batchSize))
                indices = shuffled_indices[i*self.batchSize:((i+1) * self.batchSize)]

                # load data
                minibatch_xTrain = x[indices]         # input with masking noise
                minibatch_x      = x[indices]               # original input
                minibatch_x_ori  = x_ori[indices]

                if iEpoch == 0:
                    if self.batchSize > 1:
                        minibatch_x_mean = torch.mean(minibatch_x, dim=0).unsqueeze(dim=0)
                    else:
                        minibatch_x_mean = minibatch_x

                    # calculate mean of input
                    try:
                        self.averageInput[winIdx].updateMeanStd(minibatch_x_mean.detach().clone().to('cpu'))
                    except:
                        # if the number of input changes as a result of node evolution, the counter is reinitiated
                        self.averageInput[winIdx].reset()
                        self.averageInput[winIdx].updateMeanStd(minibatch_x_mean.detach().clone().to('cpu'))

                    # calculate bias and variance
                    bias, variance, HS = self.forwardBiasVar(self.averageInput[winIdx].mean, winIdx = winIdx)

                    # update bias and variance
                    self.updateBiasVariance(bias, variance, winIdx)
                else:
                    # calculate mean of input
                    bias, variance, HS = self.forwardBiasVar(self.averageInput[winIdx].mean, winIdx = winIdx)

                    # update bias and variance
                    self.updateBiasVariance(bias, variance, winIdx)

                if evolving:
                    # growing
                    self.growNodeIdentification(bias, winIdx)
                    if self.growNode:
                        self.hiddenNodeGrowing(winIdx)
                        # print('+++Grow node TrainBasicAE+++')

                        if clustering:
                            # add an input to the cluster
                            self.ADCNcluster[winIdx].growInput(HS[-1].item())

                    # pruning
                    self.pruneNodeIdentification(variance, winIdx)
                    if self.pruneNode:
                        # print('--- Prune node TrainBasicAE---')
                        self.findLeastSignificantNode(HS)
                        self.hiddenNodePruning(winIdx)
                        evolve = 1

                        if clustering:
                            # delete an input to the cluster
                            self.ADCNcluster[winIdx].deleteInput(self.leastSignificantNode)

                            # print(self.ADCNcluster[winIdx].centroids.shape)

                # clear the gradients of all optimized variables
                optimizer = torch.optim.SGD(self.ADCNae[winIdx].network.parameters(),
                                            lr = self.lr, momentum = 0.95, weight_decay = 0.00005)
                if clustering and self.clutering_loss:
                    optimizer.add_param_group({'params': [self.ADCNcluster[winIdx].centroids]})

                optimizer.zero_grad()

                # forward
                # latentAE here is the output of the deepest autoencoder
                latentAE = self.ADCNae[winIdx].network(minibatch_xTrain) # encoder
                outputs  = self.ADCNae[winIdx].network(latentAE, 2) # decoder

                ## calculate the loss
                # reconstruction loss
                rc_loss = self.criterion(outputs, minibatch_x)

                rc_loss.backward(retain_graph=True)
                optimizer.step()
                
                if clustering and not self.growNode and self.clutering_loss:
                    # # clustering loss
                    # self.ADCNcluster[winIdx].getCluster(latentAE.to('cpu').detach().clone())
                    # oneHotClusters = F.one_hot(torch.tensor(self.ADCNcluster[winIdx].predictedClusters), num_classes =
                    # self.ADCNcluster[winIdx].nCluster).float().to(self.device)
                    # centroids      = torch.tensor(self.ADCNcluster[winIdx].centroids).float().to(self.device)
                    #
                    # # total loss
                    # loss.add_(self.regStrClusteringLoss/2,clusteringLoss(latentAE, oneHotClusters, centroids)) # change to kl

                    # kl loss
                    with torch.no_grad():
                        self.forwardADCN(minibatch_x_ori, winIdx)
                    pl = self.ADCNcluster[winIdx](self.latentFeatures)
                    ql = self.target_distribution(pl).detach()
                    kl = (torch.nn.KLDivLoss(size_average=False, reduction='mean')(F.log_softmax(pl,0), ql) / pl.shape[1])
                    kl_loss = self.alpha_kl * kl
                    if kl > 0:
                        kl_loss.backward()
                        optimizer.step()

            self.growNode  = False
            self.pruneNode = False

            if evolving and self.growNode and clustering:
                # add an input to the cluster
                _, _, HS = self.forwardBiasVar(self.averageInput[winIdx].mean, winIdx = winIdx)
                self.ADCNcluster[winIdx].growInput(HS[-1].item())

            hiddenNodeHist.append(self.nHiddenNode)

        with torch.no_grad():
            latentAE = self.ADCNae[winIdx].network(x)

        if clustering:
            self.ADCNcluster[winIdx].fit(latentAE.detach().clone().to('cpu'), epoch)

        self.hiddenNodeHist = hiddenNodeHist

        # latentAE is the extracted features of winIdx-th AE for the next hidden layer
        return latentAE.detach().clone()

    # def trainBasicAeKL(self, x, x_ori, winIdx, epoch=1, clustering=False, evolving=True, unlabeled=True):
    #     # grow layer: train CNN, train AE, train cluster
    #     # x is the extracted features from the previous hidden layer
    #     # this will update [winIdx]-th layer in greedy layer wise manner
    #     # TO DO: create grow and prune nodes, why batch size 1 is very bad, drift detection
    #     nData = x.shape[0]

    #     # masked input
    #     # maskedX = maskingNoise(x.detach().clone())              # make zero some of the input feature

    #     # prepare data
    #     x = x.to(self.device)
    #     # maskedX = maskedX.to(self.device)

    #     # get optimizer
    #     optimizer = torch.optim.SGD(self.ADCNae[winIdx].network.parameters(), lr=self.lr, momentum=0.95,
    #                                 weight_decay=0.00005)

    #     # prepare network
    #     self.ADCNae[winIdx].network = self.ADCNae[winIdx].network.train()
    #     self.ADCNae[winIdx].network = self.ADCNae[winIdx].network.to(self.device)

    #     self.growNode = False
    #     self.pruneNode = False
    #     evolve = 0
    #     hiddenNodeHist = []

    #     # print('AE training is started')

    #     for iEpoch in range(0, epoch):

    #         shuffled_indices = torch.randperm(nData)

    #         for iData in range(0, nData, self.batchSize):
    #             indices = shuffled_indices[iData:iData + self.batchSize]

    #             # load data
    #             minibatch_xTrain = x[indices]  # input with masking noise
    #             minibatch_x = x[indices]  # original input
    #             # minibatch_x_ori  = x_ori[indices]

    #             if iEpoch == 0:
    #                 if self.batchSize > 1:
    #                     minibatch_x_mean = torch.mean(minibatch_x, dim=0).unsqueeze(dim=0)
    #                 else:
    #                     minibatch_x_mean = minibatch_x

    #                 # calculate mean of input
    #                 try:
    #                     self.averageInput[winIdx].updateMeanStd(minibatch_x_mean.detach().clone().to('cpu'))
    #                 except:
    #                     # if the number of input changes as a result of node evolution, the counter is reinitiated
    #                     self.averageInput[winIdx].reset()
    #                     self.averageInput[winIdx].updateMeanStd(minibatch_x_mean.detach().clone().to('cpu'))

    #                 # calculate bias and variance
    #                 bias, variance, HS = self.forwardBiasVar(self.averageInput[winIdx].mean, winIdx=winIdx)

    #                 # update bias and variance
    #                 self.updateBiasVariance(bias, variance, winIdx)
    #             else:
    #                 # calculate mean of input
    #                 bias, variance, HS = self.forwardBiasVar(self.averageInput[winIdx].mean, winIdx=winIdx)

    #                 # update bias and variance
    #                 self.updateBiasVariance(bias, variance, winIdx)

    #             if evolving:
    #                 if self.growNode and clustering:
    #                     # add an input to the cluster
    #                     self.ADCNcluster[winIdx].growInput(HS[-1].item())

    #                 # growing
    #                 self.growNodeIdentification(bias, winIdx)
    #                 if self.growNode:
    #                     self.hiddenNodeGrowing(winIdx)
    #                     evolve = 1
    #                     print('+++ Grow node +++')

    #                 # pruning
    #                 self.pruneNodeIdentification(variance, winIdx)
    #                 if self.pruneNode:
    #                     print('--- Prune node ---')
    #                     self.findLeastSignificantNode(HS)
    #                     self.hiddenNodePruning(winIdx)
    #                     evolve = 1

    #                     # delete an input to the cluster
    #                     if clustering:
    #                         self.ADCNcluster[winIdx].deleteInput(self.leastSignificantNode)

    #             # clear the gradients of all optimized variables
    #             optimizer = torch.optim.SGD(self.ADCNae[winIdx].network.parameters(),
    #                                         lr=self.lr, momentum=0.95, weight_decay=0.00005)
    #             # optimizer.zero_grad()

    #             # forward
    #             # latentAE here is the output of the deepest autoencoder
    #             latentAE = self.ADCNae[winIdx].network(minibatch_xTrain)  # encoder
    #             outputs = self.ADCNae[winIdx].network(latentAE, 2)  # decoder

    #             # making gaussian prior
    #             eg_array = 0 + (torch.randn(np.shape(latentAE)[0], np.shape(latentAE)[1]) * 2);
    #             gaussian_prior = (eg_array - eg_array.mean(axis=0))
    #             eg_array.std(axis=0)

    #             ## calculate the loss
    #             # reconstruction loss
    #             kldivloss = nn.KLDivLoss()

    #             loss = kldivloss(latentAE, gaussian_prior)

    #             if clustering and not self.growNode:
    #                 # clustering loss
    #                 self.ADCNcluster[winIdx].getCluster(latentAE.to('cpu').detach().clone())
    #                 oneHotClusters = F.one_hot(torch.tensor(self.ADCNcluster[winIdx].predictedClusters), num_classes=
    #                 self.ADCNcluster[winIdx].nCluster).float().to(self.device)
    #                 centroids = torch.tensor(self.ADCNcluster[winIdx].centroids).float().to(self.device)

    #                 # total loss
    #                 loss.add_(self.regStrClusteringLoss / 2, clusteringLoss(latentAE, oneHotClusters, centroids))

    #             # backward pass: compute gradient of the loss with respect to model parameters
    #             # loss.backward()
    #             # perform a single optimization step (parameter update)
    #             optimizer.step()

    #         if evolving and self.growNode and clustering:
    #             # add an input to the cluster
    #             _, _, HS = self.forwardBiasVar(self.averageInput[winIdx].mean, winIdx=winIdx)
    #             self.ADCNcluster[winIdx].growInput(HS[-1].item())

    #         hiddenNodeHist.append(self.nHiddenNode)

    #     with torch.no_grad():
    #         latentAE = self.ADCNae[winIdx].network(x)

    #     if clustering:
    #         self.ADCNcluster[winIdx].fit(latentAE.detach().clone().to('cpu'), epoch)

    #     self.hiddenNodeHist = hiddenNodeHist

    #     # latentAE is the extracted features of winIdx-th AE for the next hidden layer
    #     return latentAE.detach().clone()

    # def trainBasicAeL1(self, x, x_ori, winIdx, epoch=1, clustering=False, evolving=True, unlabeled=True):
    #     # grow layer: train CNN, train AE, train cluster
    #     # x is the extracted features from the previous hidden layer
    #     # this will update [winIdx]-th layer in greedy layer wise manner
    #     # TO DO: create grow and prune nodes, why batch size 1 is very bad, drift detection
    #     nData = x.shape[0]

    #     # masked input
    #     # maskedX = maskingNoise(x.detach().clone())              # make zero some of the input feature

    #     # prepare data
    #     x = x.to(self.device)
    #     # maskedX = maskedX.to(self.device)

    #     # get optimizer
    #     optimizer = torch.optim.SGD(self.ADCNae[winIdx].network.parameters(), lr=self.lr, momentum=0.95,
    #                                 weight_decay=0.00005)

    #     # prepare network
    #     self.ADCNae[winIdx].network = self.ADCNae[winIdx].network.train()
    #     self.ADCNae[winIdx].network = self.ADCNae[winIdx].network.to(self.device)

    #     self.growNode = False
    #     self.pruneNode = False
    #     evolve = 0
    #     hiddenNodeHist = []

    #     # print('AE training is started')

    #     for iEpoch in range(0, epoch):

    #         shuffled_indices = torch.randperm(nData)

    #         for iData in range(0, nData, self.batchSize):
    #             indices = shuffled_indices[iData:iData + self.batchSize]

    #             # load data
    #             minibatch_xTrain = x[indices]  # input with masking noise
    #             minibatch_x = x[indices]  # original input


    #             if iEpoch == 0:
    #                 if self.batchSize > 1:
    #                     minibatch_x_mean = torch.mean(minibatch_x, dim=0).unsqueeze(dim=0)
    #                 else:
    #                     minibatch_x_mean = minibatch_x

    #                 # calculate mean of input
    #                 try:
    #                     self.averageInput[winIdx].updateMeanStd(minibatch_x_mean.detach().clone().to('cpu'))
    #                 except:
    #                     # if the number of input changes as a result of node evolution, the counter is reinitiated
    #                     self.averageInput[winIdx].reset()
    #                     self.averageInput[winIdx].updateMeanStd(minibatch_x_mean.detach().clone().to('cpu'))

    #                 # calculate bias and variance
    #                 bias, variance, HS = self.forwardBiasVar(self.averageInput[winIdx].mean, winIdx=winIdx)

    #                 # update bias and variance
    #                 self.updateBiasVariance(bias, variance, winIdx)
    #             else:
    #                 # calculate mean of input
    #                 bias, variance, HS = self.forwardBiasVar(self.averageInput[winIdx].mean, winIdx=winIdx)

    #                 # update bias and variance
    #                 self.updateBiasVariance(bias, variance, winIdx)

    #             if evolving:
    #                 if self.growNode and clustering:
    #                     # add an input to the cluster
    #                     self.ADCNcluster[winIdx].growInput(HS[-1].item())

    #                 # growing
    #                 self.growNodeIdentification(bias, winIdx)
    #                 if self.growNode:
    #                     self.hiddenNodeGrowing(winIdx)
    #                     evolve = 1
    #                     print('+++ Grow node +++')

    #                 # pruning
    #                 self.pruneNodeIdentification(variance, winIdx)
    #                 if self.pruneNode:
    #                     print('--- Prune node ---')
    #                     self.findLeastSignificantNode(HS)
    #                     self.hiddenNodePruning(winIdx)
    #                     evolve = 1

    #                     # delete an input to the cluster
    #                     if clustering:
    #                         self.ADCNcluster[winIdx].deleteInput(self.leastSignificantNode)

    #             # clear the gradients of all optimized variables
    #             optimizer = torch.optim.SGD(self.ADCNae[winIdx].network.parameters(),
    #                                         lr=self.lr, momentum=0.95, weight_decay=0.00005)
    #             # optimizer.zero_grad()

    #             # forward
    #             # latentAE here is the output of the deepest autoencoder
    #             latentAE = self.ADCNae[winIdx].network(minibatch_xTrain)  # encoder
    #             outputs = self.ADCNae[winIdx].network(latentAE, 2)  # decoder

    #             # making gaussian prior
    #             eg_array = 0 + (torch.randn(np.shape(outputs)[0], np.shape(outputs)[1]) * 2);
    #             gaussian_prior = (eg_array - eg_array.mean(axis=0))
    #             eg_array.std(axis=0)

    #             ## calculate the loss
    #             # reconstruction loss
    #             kldivloss = nn.KLDivLoss()

    #             loss = kldivloss(outputs, gaussian_prior)

    #             if clustering and not self.growNode:
    #                 # clustering loss
    #                 self.ADCNcluster[winIdx].getCluster(latentAE.to('cpu').detach().clone())
    #                 oneHotClusters = F.one_hot(torch.tensor(self.ADCNcluster[winIdx].predictedClusters), num_classes=
    #                 self.ADCNcluster[winIdx].nCluster).float().to(self.device)
    #                 centroids = torch.tensor(self.ADCNcluster[winIdx].centroids).float().to(self.device)

    #                 # total loss
    #                 loss.add_(self.regStrClusteringLoss / 2, clusteringLoss(latentAE, oneHotClusters, centroids))

    #             # backward pass: compute gradient of the loss with respect to model parameters
    #             # loss.backward()
    #             # perform a single optimization step (parameter update)
    #             optimizer.step()

    #         if evolving and self.growNode and clustering:
    #             # add an input to the cluster
    #             _, _, HS = self.forwardBiasVar(self.averageInput[winIdx].mean, winIdx=winIdx)
    #             self.ADCNcluster[winIdx].growInput(HS[-1].item())

    #         hiddenNodeHist.append(self.nHiddenNode)

    #     with torch.no_grad():
    #         latentAE = self.ADCNae[winIdx].network(x)

    #     if clustering:
    #         self.ADCNcluster[winIdx].fit(latentAE.detach().clone().to('cpu'), epoch)

    #     self.hiddenNodeHist = hiddenNodeHist

    #     # latentAE is the extracted features of winIdx-th AE for the next hidden layer
    #     return latentAE.detach().clone()


    def trainAE(self, x, x_ori, epoch = 1, clustering = True, evolving = True):
        # x is the extracted features from CNN encoder
        # this will update ALL layers of AE in greedy layer wise manner
        for idx, _ in enumerate(self.ADCNae):
            x = self.trainBasicAe(x, x_ori, idx, epoch = epoch, clustering = clustering, evolving = evolving)


    def trainDC(self, x, y):
        # encode decode in end-to-end manner

        # prepare network
        self.ADCNcnn = self.ADCNcnn.train()
        self.ADCNcnn = self.ADCNcnn.to(self.device)
        if self.domain_loss:
            self.discriminator = self.discriminator.train()
            self.discriminator = self.discriminator.to(self.device)

        # prepare data
        nData = x.shape[0]
        x = x.to(self.device)
        if self.domain_loss:
            y = y.to(self.device)

        # get optimizer
        optimizer = torch.optim.SGD(self.ADCNcnn.parameters(), lr=self.lr, momentum=0.95, weight_decay=0.00005)
        for iLayer, _ in enumerate(self.ADCNae):
            optimizer.add_param_group({'params': self.ADCNae[iLayer].network.parameters()})
        if self.domain_loss:
            optimizer.add_param_group({'params': self.discriminator.parameters()})

        # for loop of data batch
        shuffled_indices = torch.randperm(nData)
        batchs = int(nData/self.batchSize)

        for i in range(batchs):

            # print('Data batch:',int(iData/self.batchSize))
            indices = shuffled_indices[i*self.batchSize:((i+1) * self.batchSize)]
            minibatch_xTrain = x[indices]
            if self.domain_loss:
                domain_label = y[indices]
                # p = float(i + epoch * round(nData/self.batchSize)) / epoch / round(nData/self.batchSize)
                # alpha = 2. / (1. + np.exp(-10*p)) - 1

            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # forward encoder CNN
            feat_s = self.ADCNcnn(minibatch_xTrain)
            feat_s1 = feat_s

            # feedforward from input layer to latent space, encode
            for iLayer,_ in enumerate(self.ADCNae):
                currnet = self.ADCNae[iLayer].network
                obj     = currnet.train()
                obj     = obj.to(self.device)
                feat_s  = obj(feat_s)

            # feedforward from latent space to output layer, decode
            for iLayer in range(len(self.ADCNae)-1,0-1,-1):
                currnet = self.ADCNae[iLayer].network
                obj     = currnet.train()
                obj     = obj.to(self.device)
                feat_s  = obj(feat_s, 2)
                recons_feat_s = feat_s

            # # forward decoder CNN
            recons_x = self.ADCNcnn(feat_s, 2)

            # forward domain classifier
            domain_output = self.discriminator(feat_s1, 1.0)

            # recons loss cnn
            loss1 = self.criterion(recons_x, minibatch_xTrain) # clustering loss L1 in equ(14)
            # loss = loss1

            #recons loss AE
            loss2 = self.criterion(recons_feat_s, feat_s1)
            loss = loss1 + loss2

            # domain loss
            if self.domain_loss:
                loss3 = self.criterion(domain_output, domain_label)
                loss = loss1 + loss2 + self.alpha_dc * loss3

            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # perform a single optimization step (parameter update)
            optimizer.step()

        return x

    def trainDC_one(self, x, y):
        # encode decode in end-to-end manner

        # prepare data
        shuffled_indices = torch.randperm(np.shape(x)[0])
        img_x = x[shuffled_indices]
        x  =  x[shuffled_indices]
        if self.domain_loss:
            domain_label = y[shuffled_indices]

        # prepare network
        self.ADCNcnn = self.ADCNcnn.train()
        self.ADCNcnn = self.ADCNcnn.to(self.device)
        if self.domain_loss:
            self.discriminator = self.discriminator.train()
            self.discriminator = self.discriminator.to(device)

        # get optimizer
        optimizer = torch.optim.SGD(self.ADCNcnn.parameters(), lr=self.lr, momentum=0.95, weight_decay=0.00005)
        for iLayer, _ in enumerate(self.ADCNae):
            optimizer.add_param_group({'params': self.ADCNae[iLayer].network.parameters()})
        if self.domain_loss:
            optimizer.add_param_group({'params': self.discriminator.parameters()})

        # clear the gradients of all optimized variables
        optimizer.zero_grad()

        # forward encoder CNN
        x = self.ADCNcnn(x)
        feat_s = x

        # feedforward from input layer to latent space, encode
        for iLayer,_ in enumerate(self.ADCNae):
            currnet = self.ADCNae[iLayer].network
            obj     = currnet.train()
            obj     = obj.to(self.device)
            x  = obj(x)

        # feedforward from latent space to output layer, decode
        for iLayer in range(len(self.ADCNae)-1,0-1,-1):
            currnet = self.ADCNae[iLayer].network
            obj     = currnet.train()
            obj     = obj.to(self.device)
            x       = obj(x, 2)
            recons_feat_s = x

        # # forward decoder CNN
        recons_x = self.ADCNcnn(x, 2)

        # forward domain classifier
        domain_output = self.discriminator(feat_s, 1.0)

        # recons loss cnn
        loss1 = self.criterion(recons_x, img_x) # clustering loss L1 in equ(14)
        # loss = loss1

        #recons loss AE
        loss2 = self.criterion(recons_feat_s, feat_s)
        loss = loss1 + loss2

        # domain loss
        if self.domain_loss:
            loss3 = self.criterion(domain_output, domain_label)
            loss = loss1 + loss2 + self.alpha_dc * loss3
            # print(loss3)

        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # perform a single optimization step (parameter update)
        optimizer.step()

        return x


    def train_main(self, batchDataS, batchDataT, nEpochPreq=1, nEpochKLL1=1):
        # train ADCN, all network structures, layer wise manner
        x = torch.cat((batchDataS,batchDataT),dim=0)
        extractedFeatures = self.trainCNN(x, unlabeled = True, epoch = nEpochPreq)
        self.trainAE(extractedFeatures, x, epoch = nEpochPreq, evolving = self.evolving)

        # domain label (0-source data 1-target data)
        if self.domain_loss:
            sourcelabel = torch.zeros(batchDataS.shape[0])
            targetlabel = torch.ones(batchDataT.shape[0])
            domain_label = torch.cat((sourcelabel, targetlabel),0)
            domain_label = domain_label.long()
            domain_label = torch.zeros(x.shape[0], 2).scatter_(1, domain_label.unsqueeze(1), 1)
        else:
            domain_label = None
    
        for iEpoch in range(0, nEpochKLL1):
            # print('=======Epoch==========TrainDomain:', iEpoch+1)
            self.trainDC(x, domain_label)
            # self.trainDC_one(x, domain_label)

    # def fitLeo(self, batchDataS, batchDataT, nEpochPreq = 1, nEpochKLL1=1):

    #     # train ADCN, all network structures, layer wise manner
    #     # x is image data with size 1 x 28 x 28
    #     x = torch.cat((batchDataS,batchDataT),dim=0)
    #     self.optimizeCNN(x) # there is a random function already to shuffle batchDataS and batchDataT
    #     extractedFeatures = self.trainCNN(x, unlabeled = True, epoch = nEpochPreq)#initialization 50 epoch, data sudah gabungan target and source
    #     self.trainAE(extractedFeatures, x, epoch = nEpochPreq, evolving = self.evolving)
    #     # recons loss cnn

    #     for i in range(0,nEpochKLL1):
    #         print("epoch for KL and L1 :",i+1, "  from ",nEpochKLL1)
    #         self.optimizeSource(batchDataS)
    #         self.optimizeTarget(batchDataT)


    #     # self.fitLeopard(batchDataS, batchDataT, winIdx=0,epoch=epoch, clustering=True, evolving=True)
    #     #self.trainBasicAeL1(extractedFeatures,batchDataT, winIdx=0,epoch=epoch, evolving=False)

    def fitCL(self, x, reconsLoss = False, unlabeled = True, epoch = 1):
        # x is current task image data with size 1 x 28 x 28
        # train ADCN to retain the knowledge in old task, part of continual learning
        # executed iff there is at least 1 old task
        if len(self.ADCNold) > 0:
            nData = x.shape[0]
            x     = x.to(self.device)

            # get optimizer
            optimizer = torch.optim.SGD(self.ADCNcnn.parameters(), lr = self.lr, momentum = 0.95, weight_decay = 0.00005)
            for iLayer,_ in enumerate(self.ADCNae):
                optimizer.add_param_group({'params': self.ADCNae[iLayer].network.parameters()})

            # prepare network
            self.ADCNcnn = self.ADCNcnn.train()
            self.ADCNcnn = self.ADCNcnn.to(self.device)

            # print('LWF training is started')

            for iEpoch in range(0, epoch):

                # if iEpoch%20 == 0:
                #     print('Epoch: ', iEpoch)

                shuffled_indices = torch.randperm(nData)
                batchs = int(nData/self.batchSize)

                for i in range(batchs):

                    # print('Data batch:',int(iData/self.batchSize))
                    indices = shuffled_indices[i*self.batchSize:((i+1) * self.batchSize)]

                    # load data
                    minibatch_xTrain = x[indices]

                    # clear the gradients of all optimized variables
                    optimizer.zero_grad()

                    # forward
                    outputs = self.forward(minibatch_xTrain)

                    # calculate the LWF loss, accross all previous task
                    loss    = self.LwFloss(outputs, minibatch_xTrain)
                    lwfLoss = loss.detach().clone().item()

                    if reconsLoss:
                        lossRecons = self.criterion(outputs, minibatch_xTrain)
                        loss.add_(lossRecons)

                    # backward pass: compute gradient of the loss with respect to model parameters
                    loss.backward()

                    # perform a single optimization step (parameter update)
                    optimizer.step()
            print('LWF loss: ', lwfLoss, 'recons Loss: ',loss.detach().clone().item() - lwfLoss)

    def updateAllegiance(self, labeledData, labeledLabel, randomTesting = False):
        # At the end of each phase, we introduce a limited amount of labeled data per class to evaluate classification accuracy.
        # forward to each layer
        for iLayer in range(len(self.ADCNae)):
            self.forwardADCN(labeledData, winIdx = iLayer)

            # update allegiance in each layer
            self.ADCNcluster[iLayer].updateAllegiance(self.latentFeatures.detach().clone().to('cpu'),
                                                      labeledLabel.to('cpu'), randomTesting = randomTesting)

    # ============================= Testing =============================
    def predict(self, x):
        with torch.no_grad():
            x = x.to(self.device)

            # prepare network
            self.ADCNcnn = self.ADCNcnn.eval()
            self.ADCNcnn = self.ADCNcnn.to(self.device)

            # forward encoder CNN
            x = self.ADCNcnn(x)

            # feedforward from input layer to latent space
            score = np.zeros([x.shape[0], self.nOutput])
            for iLayer in range(len(self.ADCNae)):
                currnet = self.ADCNae[iLayer].network
                obj     = currnet.eval()
                obj     = obj.to(self.device)
                x       = obj(x)

                # predict
                self.ADCNcluster[iLayer].predict(x.detach().clone().to('cpu'))
                # pdb.set_trace()
                score = score + stableSoftmax(self.ADCNcluster[iLayer].score.numpy())

        self.predictedLabel = np.argmax(score,1)
        self.score          = score

    def testing(self, x, label):
        # testing
        start_test          = time.time()
        self.predict(x)
        end_test            = time.time()
        self.testingTime    = end_test - start_test
        correct             = (self.predictedLabel == label.numpy()).sum().item()
        self.accuracy       = 100*correct/(self.predictedLabel == label.numpy()).shape[0]  # 1: correct, 0: wrong
        self.trueClassLabel = label.numpy()

    def randomTesting(self, x, label):
        # testing
        start_test          = time.time()
        self.predict(x)
        end_test            = time.time()
        self.testingTime    = end_test - start_test
        randomPrediction    = np.random.randint(self.nOutput, size=self.predictedLabel.shape[0])
        correct             = (randomPrediction == label.numpy()).sum().item()
        self.accuracy       = 100*correct/(self.predictedLabel == label.numpy()).shape[0]  # 1: correct, 0: wrong
        self.trueClassLabel = label.numpy()

    # ============================= Evolving mechanism =============================
    def layerGrowing(self):
        if len(self.ADCNae) == len(self.ADCNcluster):
            print('Add a new layer:[{}, {}]'.format(self.ADCNae[-1].nNodes,int(self.ADCNae[-1].nNodes/2)))
            self.ADCNae       = self.ADCNae + [smallAE(self.ADCNae[-1].nNodes,int(self.ADCNae[-1].nNodes/2))]
            self.averageBias  = self.averageBias  + [meanStdCalculator()]
            self.averageVar   = self.averageVar   + [meanStdCalculator()]
            self.averageInput = self.averageInput + [meanStdCalculator()]
            self.nHiddenLayer = len(self.ADCNae)
            print('*** ADD a new LAYER ***')

    def hiddenNodeGrowing(self, winIdx):
        if winIdx <= (len(self.ADCNae)-1):
            copyNet = copy.deepcopy(self.ADCNae[winIdx])
            copyNet.nodeGrowing(device = self.device)
            self.ADCNae[winIdx] = copy.deepcopy(copyNet)
            if winIdx != (len(self.ADCNae)-1):
                copyNextNet = copy.deepcopy(self.ADCNae[winIdx+1])
                copyNextNet.inputGrowing(device = self.device)
                self.ADCNae[winIdx+1] = copy.deepcopy(copyNextNet)

            # print('+++ GROW a hidden NODE +++')
            self.updateNetProperties()
        else:
            raise IndexError

    def hiddenNodePruning(self, winIdx):
        if winIdx <= (len(self.ADCNae)-1):
            copyNet = copy.deepcopy(self.ADCNae[winIdx])
            copyNet.nodePruning(self.leastSignificantNode)
            self.ADCNae[winIdx] = copy.deepcopy(copyNet)
            if winIdx != (len(self.ADCNae)-1):
                copyNextNet = copy.deepcopy(self.ADCNae[winIdx+1])
                copyNextNet.inputPruning(self.leastSignificantNode)
                self.ADCNae[winIdx+1] = copy.deepcopy(copyNextNet)

            # print('--- Hidden NODE No: ',self.leastSignificantNode,' is PRUNED ---')
            self.updateNetProperties()
        else:
            raise IndexError

    # ============================= Network Evaluation =============================
    def LwFloss(self, currentBatchOutput, currentBatchData):
        loss = []
        criterion = nn.BCELoss()
        for iTask,_ in enumerate(self.ADCNold):
            with torch.no_grad():
                minibatch_xOld = self.ADCNold[iTask].forward(currentBatchData, self.device)     # it acts as the target

            regStr = self.regStrLWF*(1.0 - self.nOutputPerTask/((iTask + 1)*self.nOutputPerTask + self.nOutputPerTask))
            loss.append(regStr*criterion(currentBatchOutput, minibatch_xOld.detach().clone()))

        return sum(loss)

    def driftDetection(self, batchData, prevBatchData = None):
        self.ADCNcnn = self.ADCNcnn.to(self.device)
        batchData    = batchData.to(self.device)

        with torch.no_grad():
            # forward encoder CNN
            currFeatureMatrix = self.ADCNcnn(batchData)
            currFeatureMatrix = currFeatureMatrix.to('cpu')

            if prevBatchData is not None:
                prevBatchData = prevBatchData.to(self.device)

                # forward encoder CNN
                prevFeatureMatrix = self.ADCNcnn(prevBatchData)
                prevFeatureMatrix = prevFeatureMatrix.to('cpu')
            # currFeatureMatrix is a list containing the mean of extracted features

        self.driftStatusOld = self.driftStatus
        driftStatus = 0  # 0: no drift, 1: warning, 2: drift

        if self.driftStatusOld != 2:
            # detect drift
            # combine buffer data, when previous batch is warning
            if self.driftStatusOld == 1:
                with torch.no_grad():
                    bufferFetureMatrix = self.bufferFetureMatrix.to(self.device)

                    # forward encoder CNN
                    bufferFetureMatrix = self.ADCNcnn(bufferFetureMatrix)
                    bufferFetureMatrix = bufferFetureMatrix.to('cpu')

                    currFeatureMatrix = torch.cat((bufferFetureMatrix,currFeatureMatrix),0)

            # combine current and previous feature matrix
            combinedFeatureMatrix = currFeatureMatrix
            if prevBatchData is not None:
                combinedFeatureMatrix = torch.cat((prevFeatureMatrix,currFeatureMatrix),0)

            # prepare statistical coefficient to confirm a cut point
            nData             = combinedFeatureMatrix.shape[0]
            cutPointCandidate = [int(nData/4),int(nData/2),int(nData*3/4)]
            cutPoint          = 0
            miu_X             = torch.mean(combinedFeatureMatrix,0)    # statistics of the current batch data extracted features
            errorBoundX       = np.sqrt((1/(2*nData))*np.log(1/self.alphaDrift))

            # confirm the cut point
            for iCut in cutPointCandidate:
                miu_V       = torch.mean(combinedFeatureMatrix[0:iCut], 0)
                nV          = combinedFeatureMatrix[0:iCut].shape[0]
                errorBoundV = np.sqrt((1/(2*nV))*np.log(1/self.alphaDrift))
                if torch.mean(miu_X + errorBoundX).item() <= torch.mean(miu_V + errorBoundV).item():
                    cutPoint = iCut
                    print('A cut point is detected cut: ', cutPoint)
                    break

            # confirm drift
            # if np.abs(miu_F - miu_E) >= errorBoundDrift:   # This formula is able to detect drift, even the performance improves
            if cutPoint > 0:
                # prepare statistical coefficient to confirm a drift
                max_b,_ = torch.max(combinedFeatureMatrix, 0)
                min_a,_ = torch.min(combinedFeatureMatrix, 0)
                diff_ba = max_b - min_a

                errorBoundDrift = torch.mean(diff_ba*np.sqrt(((nData - nV)/(2*nV*nData))*np.log(1/self.alphaDrift))).item()
                # if miu_V - miu_X >= errorBoundDrift:   # This formula is only able to detect drift when the performance decreses
                # print('np.abs(miu_V - miu_X) : ', np.abs(miu_V - miu_X),'>','errorBoundDrift',errorBoundDrift)

                if torch.mean(np.abs(miu_V - miu_X)).item() >= errorBoundDrift:   # This formula is able to detect drift, even the performance improves
                    # confirm drift
                    print('H0 is rejected with size: ', errorBoundDrift)
                    print('Status: DRIFT')
                    driftStatus             = 2
                    self.bufferFetureMatrix = []
                    # self.prevFeatureMatrix  = []

                else:
                    # prepare statistical coefficient to confirm a warning
                    errorBoundWarning = torch.mean(diff_ba*np.sqrt(((nData - nV)/(2*nV*nData))*np.log(1/self.alphaWarning))).item()
                    # print('np.abs(miu_V - miu_X) : ', np.abs(miu_V - miu_X),'>','errorBoundWarning',errorBoundWarning)
                    # if there is a warning in the previous batch, then there is only 2 option left: drift or stable.
                    # it is assumed that the number of samples is large enough to confirm those 2 situation
                    # if miu_V - miu_X >= errorBoundWarning and self.driftStatusOld != 1:
                    if torch.mean(np.abs(miu_V - miu_X)).item() >= errorBoundWarning and self.driftStatusOld != 1:
                        # confirm warning
                        # if there is a warning, the currFeatureMatrix is stored in the buffer and will be evaluated in the next batch
                        # together with the currFeatureMatrix of the next batch
                        print('H0 is rejected with size: ', errorBoundWarning)
                        print('Status: WARNING')
                        driftStatus = 1
                        self.bufferFetureMatrix = prevBatchData.to('cpu')
                    else:
                        # confirm stable
                        print('H0 is NOT rejected, size:', torch.mean(np.abs(miu_V - miu_X)).item(),
                              '; Error bound warning: ', errorBoundWarning, '; Error bound drift: ', errorBoundDrift)
                        print('Status: STABLE')
                        driftStatus = 0
                        self.bufferFetureMatrix = []
                        # self.prevFeatureMatrix  = currFeatureMatrix
            else:
                # there is no cutpoint detected means that there is no significant increase in the combinedFeatureMatrix
                print('Cut point is NOT detected')
                print('Status: STABLE')
                driftStatus = 0
                # self.prevFeatureMatrix = currFeatureMatrix

        self.driftStatus = driftStatus
        self.driftHistory.append(driftStatus)

    def growNodeIdentification(self, bias, winIdx):
        # confirm high variance situation
        # winIdx is the indes of current AE network, started from 0 (np.log2(winIdx) + 1)
        winIdxa           = winIdx + 1       # winIdx start from 0, that is why it is required to add 1
        # dynamicKsigmaGrow = (np.log(winIdxa) + 1)*(1.3*np.exp(-bias) + 0.7)
        dynamicKsigmaGrow = (1.3*np.exp(-bias) + 0.7)
        # dynamicKsigmaGrow = (1.2*np.exp(-bias) + 0.8)
        growCondition1    = (self.averageBias[winIdx].minMean +
                             dynamicKsigmaGrow*self.averageBias[winIdx].minStd)
        growCondition2    = self.averageBias[winIdx].mean + self.averageBias[winIdx].std

        # print('growCondition2: ', growCondition2,'growCondition1: ', growCondition1)
        if growCondition2 > growCondition1 and self.averageBias[winIdx].count >= 1:
            self.growNode = True
            # if winIdx > 0:
            #     pdb.set_trace()
        else:
            self.growNode = False

    def pruneNodeIdentification(self, variance, winIdx):
        # confirm high variance situation
        # winIdx is the indes of current AE network, started from 0
        winIdxa            = winIdx + 1      # winIdx start from 0, that is why it is required to add 1
        # dynamicKsigmaPrune = (np.log(winIdxa) + 1)*(1.3*np.exp(-variance) + 0.7)
        dynamicKsigmaPrune = (1.3*np.exp(-variance) + 0.7)
        pruneCondition1    = (self.averageVar[winIdx].minMean +
                              2*dynamicKsigmaPrune*self.averageVar[winIdx].minStd)
        pruneCondition2    = self.averageVar[winIdx].mean + self.averageVar[winIdx].std

        if (pruneCondition2 > pruneCondition1 and not self.growNode and
                self.averageVar[winIdx].count >= 20 and
                self.ADCNae[winIdx].nNodes > self.nOutput):
            self.pruneNode = True
        else:
            self.pruneNode = False

    def findLeastSignificantNode(self, hiddenNodeSignificance):
        # find the least significant node given the hidden node significance in the current layer, only for AE network
        # should be executed after doing feedforwardBiasVar on the winning layer
        self.leastSignificantNode = torch.argmin(torch.abs(hiddenNodeSignificance)).tolist()
        # print('Pruned node: ', self.leastSignificantNode)

    def updateBiasVariance(self, bias, variance, winIdx):
        # calculate mean of bias and variance
        # should be executed after doing feedforwardBiasVar on the winning layer
        self.averageBias[winIdx].updateMeanStd(bias)
        if self.averageBias[winIdx].count < 1 or self.growNode:
            self.averageBias[winIdx].resetMinMeanStd()
            # if winIdx > 0:
            #     pdb.set_trace()
        else:
            self.averageBias[winIdx].updateMeanStdMin()

        # calculate mean of variance
        self.averageVar[winIdx].updateMeanStd(variance)
        if self.averageVar[winIdx].count < 20 or self.pruneNode:
            self.averageVar[winIdx].resetMinMeanStd()
        else:
            self.averageVar[winIdx].updateMeanStdMin()

    # ============================= Data management =============================
    def trainingDataPreparation(self, batchData, batchLabel):
        # training data preparation
        # if there is a warning, the data is stored in the buffer, and no training is conducted
        # if there is a drift or a stable condition and the buffer is not empty, then the data in the
        # buffer is concatenated together with the current data
        if self.driftStatus == 0 or self.driftStatus == 2:  # STABLE or DRIFT
            # check buffer
            if self.bufferData.shape[0] != 0:
                # add buffer to the current data batch
                self.batchData  = torch.cat((self.bufferData,batchData),0)
                self.batchLabel = torch.cat((self.bufferLabel,batchLabel),0)

                # clear buffer
                self.bufferData  = torch.Tensor().float()
                self.bufferLabel = torch.Tensor().long()

        if self.driftStatus == 1:  # WARNING
            # store data to buffer
            # print('Store data to buffer')
            self.bufferData  = batchData
            self.bufferLabel = batchLabel

        return batchData, batchLabel

    def augmentLabeledSamples(self, data, labels):
        # augment the collection of labeled samples for the network and clusters
        # should be executed after training
        self.labeledData  = torch.cat((self.labeledData,data),0)
        self.labeledLabel = torch.cat((self.labeledLabel,labels),0)

        ## augment input data for each cluster
        # prepare model
        self.ADCNcnn = self.ADCNcnn.to(self.device)

        # forward encoder CNN
        data = data.to(self.device)

        with torch.no_grad():

            data = self.ADCNcnn(data)

            # feedforward from input layer to latent space
            for iLayer in range(len(self.ADCNae)):
                currnet = self.ADCNae[iLayer].network
                obj     = currnet.train()
                obj     = obj.to(self.device)
                data    = obj(data)

                self.ADCNcluster[iLayer].augmentLabeledSamples(data.detach().clone().to('cpu').numpy(), labels.to('cpu').numpy())

    def storeOldModel(self, taskId):
        # store taskId-th task model to generate output from the network of the previous task
        print('store model : ', taskId)

        # create blank network
        oldNet = ADCNoldtask(taskId)

        # copy net property
        oldNet.nInput       = self.nInput
        oldNet.nOutput      = self.nOutput
        oldNet.nHiddenLayer = self.nHiddenLayer
        oldNet.nHiddenNode  = self.nHiddenNode
        oldNet.taskId       = taskId

        # copy network
        oldNet.ADCNcnn      = copy.deepcopy(self.ADCNcnn)
        oldNet.ADCNae       = copy.deepcopy(self.ADCNae)

        # put it in the collection of old task
        self.ADCNold  = self.ADCNold  + [oldNet]

class ADCNMPL(nn.Module):
    def __init__(self, nOutput, nInput = 196, nHiddenNode = 96, alpha_w = 0.005, alpha_d = 0.001, 
    LR = 0.01, desiredLabels = [1,2,3,4,5], alpha_kl = 0.0, alpha_dc = 0.0):
        super(ADCNMPL,self).__init__()
        # random seed control
        # np.random.seed(0)
        # torch.manual_seed(0)
        # random.seed(0)

        # initial network
        self.ADCNcnn     = ConvAeMNIST()
        self.ADCNae      = [smallAE(nInput, nHiddenNode)]
        self.discriminator = domain_classifier(nInput, nHiddenNode)
        self.ADCNold     = []
        self.ADCNcluster = []

        # network significance
        self.averageBias  = [meanStdCalculator()]
        self.averageVar   = [meanStdCalculator()]
        self.averageInput = [meanStdCalculator()]

        # hyperparameters
        self.lr        = LR
        self.criterion = nn.MSELoss()

        # drift detection parameters
        self.alphaWarning      = alpha_w
        self.alphaDrift        = alpha_d
        self.driftStatusOld    = 0
        self.driftStatus       = 0
        self.driftHistory      = []
        # self.prevFeatureMatrix = []
        self.bufferData  = torch.Tensor().float()
        self.bufferLabel = torch.Tensor().long()

        # Evolving
        self.growNode       = False
        self.pruneNode      = False
        self.evolving       = True
        self.clusterGrowing = True

        # net properties
        self.nInput       = nInput
        self.nOutput      = nOutput
        self.nHiddenLayer = 1
        self.nHiddenNode  = nHiddenNode

        # cluster properties
        self.desiredLabels= desiredLabels
        self.nInitCluster = 2
        self.nCluster     = self.nInitCluster
        self.regStrClusteringLoss = 0.01

        # LWF
        self.nOutputPerTask = nOutput
        self.regStrLWF      = 5

        # params
        self.clutering_loss  = True
        self.domain_loss     = True
        self.mmd_d           = False
        self.alpha_kl        = alpha_kl
        self.alpha_dc        = alpha_dc
        self.mmd             = False
        if self.alpha_kl == 0.0:
            self.clutering_loss = False
        if self.alpha_dc == 0.0:
            self.domain_loss = False

    def updateNetProperties(self):
        self.nHiddenLayer = len(self.ADCNae)
        nHiddenNode = 0
        nCluster    = 0
        for idx, nett in enumerate(self.ADCNae):
            nHiddenNode  += nett.nNodes
            try:
                nCluster   += self.ADCNcluster[idx].nCluster
            except:
                nCluster    = 0

        self.nHiddenNode  = nHiddenNode
        self.nCluster     = nCluster

    # def getNetProperties(self):
    #     for _,nett in enumerate(self.ADCNae):
    #         nett.getNetProperties()

    @staticmethod
    def kld(p, q):
        res = torch.sum(p * torch.log(p / q), dim=-1)
        return res

    @staticmethod
    def target_distribution(x):
        weight = (x ** 2) / torch.sum(x, 0)
        return (weight.t() / torch.sum(weight,1)).t()

    # ============================= forward pass =============================
    def forwardADCN(self, x, winIdx = None):
        # prepare model
        self.ADCNcnn = self.ADCNcnn.to(self.device)

        # forward encoder CNN
        x = x.to(self.device)
        x = self.ADCNcnn(x)

        # feedforward from input layer to latent space
        for iLayer in range(len(self.ADCNae)):
            currnet = self.ADCNae[iLayer].network
            obj     = currnet.train()
            obj     = obj.to(self.device)
            x       = obj(x)

            if winIdx is not None:
                # store latent features of the winning layer for clustering
                if iLayer == winIdx:
                    self.latentFeatures = x.clone()

                if iLayer == len(self.ADCNae) - 2:
                    # store input feature for the last hidden layer
                    self.lastInputFeature = x.clone()

        return x

    def forwardBiasVar(self, target, winIdx):
        # x is the input features for this layer
        # only for ADCNae
        with torch.no_grad():
            target = target.to(self.device)

            # encode
            x  = self.ADCNae[winIdx].network(target)
            hiddenNodeSignificance = torch.mean(x.detach().clone(), 0)
            x2 = (x.detach().clone())**2

            # decode
            x  = self.ADCNae[winIdx].network(x, 2)
            x2 = self.ADCNae[winIdx].network(x2, 2)

            # get bias and variance
            bias     = torch.mean((x - target)**2).item()
            variance = torch.mean(x2 - target**2).item()
            # pdb.set_trace()

        return bias, variance, hiddenNodeSignificance

    # def forward(self, x):
    #     # encode decode in end-to-end manner
    #     # prepare model
    #     self.ADCNcnn = self.ADCNcnn.to(self.device)
    #
    #     # prepare data
    #     x = x.to(self.device)
    #
    #     # forward encoder CNN
    #     x = self.ADCNcnn(x)
    #
    #     # feedforward from input layer to latent space, encode
    #     for iLayer,_ in enumerate(self.ADCNae):
    #         currnet = self.ADCNae[iLayer].network
    #         obj     = currnet.train()
    #         obj     = obj.to(self.device)
    #         x       = obj(x)
    #
    #         # should consider clustering loss
    #
    #
    #     # feedforward from latent space to output layer, decode
    #     for iLayer in range(len(self.ADCNae)-1,0-1,-1):
    #         currnet = self.ADCNae[iLayer].network
    #         obj     = currnet.train()
    #         obj     = obj.to(self.device)
    #         x       = obj(x, 2)
    #
    #     # forward decoder CNN
    #     x = self.ADCNcnn(x, 2)
    #
    #     return x

    def optimizeCNN(self, x):
        # encode decode in end-to-end manner
        # prepare model

        shuffled_indices = torch.randperm(np.shape(x)[0])
        img_x = x[shuffled_indices]
        x = x[shuffled_indices]
        self.ADCNcnn = self.ADCNcnn.to(self.device)

        # get optimizer
        optimizer = torch.optim.SGD(self.ADCNcnn.parameters(), lr=self.lr, momentum=0.95, weight_decay=0.00005)
        for iLayer, _ in enumerate(self.ADCNae):
            optimizer.add_param_group({'params': self.ADCNae[iLayer].network.parameters()})

        # prepare network
        self.ADCNcnn = self.ADCNcnn.train()
        self.ADCNcnn = self.ADCNcnn.to(self.device)

        # prepare data
        x = x.to(self.device)

        # forward encoder CNN
        x = self.ADCNcnn(x)

        # feedforward from input layer to latent space, encode
        for iLayer,_ in enumerate(self.ADCNae):
            currnet = self.ADCNae[iLayer].network
            obj     = currnet.train()
            obj     = obj.to(self.device)
            x       = obj(x)

            # should consider clustering loss


        # feedforward from latent space to output layer, decode
        for iLayer in range(len(self.ADCNae)-1,0-1,-1):
            currnet = self.ADCNae[iLayer].network
            obj     = currnet.train()
            obj     = obj.to(self.device)
            x       = obj(x, 2)

        # forward decoder CNN
        recons_x = self.ADCNcnn(x, 2)

        loss = self.criterion(recons_x, img_x)

        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # perform a single optimization step (parameter update)
        optimizer.step()

        return x

    # def optimizeSource(self, x, epoch, n_epoch):
    #     # encode decode in end-to-end manner
    #     # prepare model
    #     Tensor = torch.FloatTensor
    #     nData = np.shape(x)[0]

    #     self.ADCNcnn = self.ADCNcnn.to(self.device)
    #     # prepare data
    #     x = x.to(self.device)

    #     # get optimizer
    #     optimizer = torch.optim.SGD(self.ADCNcnn.parameters(), lr=self.lr, momentum=0.95, weight_decay=0.00005)
    #     for iLayer, _ in enumerate(self.ADCNae):
    #         optimizer.add_param_group({'params': self.ADCNae[iLayer].network.parameters()})
    #     optimizer.add_param_group({'params': self.discriminator.parameters()})

    #     # prepare network
    #     self.ADCNcnn = self.ADCNcnn.train()
    #     self.ADCNcnn = self.ADCNcnn.to(self.device)

    #     # for loop of data batch
    #     shuffled_indices = torch.randperm(nData)

    #     for i, iData in enumerate(range(0,nData,self.batchSize)):
    #         # load data
    #         indices = shuffled_indices[iData:iData + self.batchSize]
    #         minibatch_xTrain = x[indices]

    #         p = float(i + epoch * round(nData/self.batchSize)) / n_epoch / round(nData/self.batchSize)
    #         alpha = 2. / (1. + np.exp(-10*p)) - 1

    #         batchsize = minibatch_xTrain.shape[0]
    #         domain_label = torch.zeros(batchsize)
    #         domain_label = domain_label.long()
    #         domain_label = torch.zeros(batchsize,2).scatter_(1,domain_label.unsqueeze(1),1)
    #         domain_label = domain_label[shuffled_indices]

    #         # clear the gradients of all optimized variables
    #         optimizer.zero_grad()

    #         # forward encoder CNN
    #         feat_s = self.ADCNcnn(minibatch_xTrain)
    #         feat_s1 = feat_s

    #         # feedforward from input layer to latent space, encode
    #         for iLayer,_ in enumerate(self.ADCNae):
    #             currnet = self.ADCNae[iLayer].network
    #             obj     = currnet.train()
    #             obj     = obj.to(self.device)
    #             feat_s  = obj(feat_s)
    #             latentAE_s = feat_s

    #         # feedforward from latent space to output layer, decode
    #         for iLayer in range(len(self.ADCNae)-1,0-1,-1):
    #             currnet = self.ADCNae[iLayer].network
    #             obj     = currnet.train()
    #             obj     = obj.to(self.device)
    #             feat_s  = obj(feat_s, 2)
    #             recons_feat_s = feat_s

    #         # # forward decoder CNN
    #         recons_x = self.ADCNcnn(feat_s, 2)

    #         # forward domain classifier
    #         domain_output = self.discriminator(feat_s1, alpha)

    #         # recons loss cnn
    #         loss = self.criterion(recons_x, minibatch_xTrain) # clustering loss L1 in equ(14)

    #         #recons loss AE
    #         loss.add_(self.criterion(recons_feat_s, feat_s1))

    #         # domain loss
    #         print(self.criterion(domain_output, domain_label))
    #         loss.add_(self.criterion(domain_output, domain_label))

    #         # #loss KL to align the prior to gaussian prior
    #         # Alpha = 0.01
    #         #
    #         # # gaussian prior source
    #         # zn_s = torch.normal(0, 1, size=(np.shape(latentAE_s)[0], np.shape(latentAE_s)[1]))
    #         #
    #         # # loss.add_(Alpha,F.kl_div(latentAE_s,zn_s))
    #         # kl = F.kl_div(torch.softmax(latentAE_s, dim=1), torch.log_softmax(zn_s, dim=1),
    #         #                           reduction='none').mean()
    #         # # print(kl,kl.requires_grad)
    #         # loss = Alpha * kl # equ (19) requires_grad

    #         # backward pass: compute gradient of the loss with respect to model parameters
    #         loss.backward()

    #         # perform a single optimization step (parameter update)
    #         optimizer.step()

    #     return x

    # def optimizeTarget(self, x, epoch, n_epoch):
    #     # encode decode in end-to-end manner
    #     # prepare model
    #     Tensor = torch.FloatTensor
    #     nData = np.shape(x)[0]

    #     self.ADCNcnn = self.ADCNcnn.to(self.device)
    #     criterionDAL = nn.L1Loss().cpu()

    #     # get optimizer
    #     optimizer = torch.optim.SGD(self.ADCNcnn.parameters(), lr=self.lr, momentum=0.95, weight_decay=0.00005)
    #     for iLayer, _ in enumerate(self.ADCNae):
    #         optimizer.add_param_group({'params': self.ADCNae[iLayer].network.parameters()})
    #     optimizer.add_param_group({'params': self.discriminator.parameters()})

    #     # prepare network
    #     self.ADCNcnn = self.ADCNcnn.train()
    #     self.ADCNcnn = self.ADCNcnn.to(self.device)

    #     # prepare data
    #     x = x.to(self.device)

    #     # for loop of data batch
    #     shuffled_indices = torch.randperm(nData)

    #     for i, iData in enumerate(range(0,nData,self.batchSize)):
    #         # load data
    #         indices = shuffled_indices[iData:iData + self.batchSize]
    #         minibatch_xTrain = x[indices]

    #         p = float(i + epoch * round(nData/self.batchSize)) / n_epoch / round(nData/self.batchSize)
    #         alpha = 2. / (1. + np.exp(-10*p)) - 1

    #         batchsize = minibatch_xTrain.shape[0]
    #         domain_label = torch.ones(batchsize)
    #         domain_label = domain_label.long()
    #         domain_label = torch.zeros(batchsize,2).scatter_(1,domain_label.unsqueeze(1),1)

    #         # clear the gradients of all optimized variables
    #         optimizer.zero_grad()

    #         # forward encoder CNN
    #         feat = self.ADCNcnn(minibatch_xTrain)
    #         feat_t = feat

    #         # feedforward from input layer to latent space, encode
    #         for iLayer,_ in enumerate(self.ADCNae):
    #             currnet = self.ADCNae[iLayer].network
    #             obj     = currnet.train()
    #             obj     = obj.to(self.device)
    #             feat  = obj(feat)
    #             latentAE_t = feat

    #             # should consider clustering loss

    #         # feedforward from latent space to output layer, decode
    #         for iLayer in range(len(self.ADCNae)-1,0-1,-1):
    #             currnet = self.ADCNae[iLayer].network
    #             obj     = currnet.train()
    #             obj     = obj.to(self.device)
    #             feat    = obj(feat, 2)
    #             recons_feat_t = feat

    #         # forward decoder CNN
    #         recons_x = self.ADCNcnn(feat, 2)

    #         # forward domain classifier
    #         domain_output = self.discriminator(feat_t, alpha)

    #         #recons loss cnn
    #         loss = self.criterion(recons_x, minibatch_xTrain)

    #         # #recons loss AE
    #         loss.add_(self.criterion(recons_feat_t, feat_t))

    #         # domain loss
    #         print(self.criterion(domain_output, domain_label))
    #         loss.add_(self.criterion(domain_output, domain_label))

    #         # #loss KL to align the prior to gaussian prior
    #         # Alpha = 0.01
    #         # # gaussian prior source
    #         # zn_t =torch.normal(0, 1, size=(np.shape(latentAE_t)[0], np.shape(latentAE_t)[1]))
    #         #
    #         # with torch.no_grad():
    #         #     # feedforward from latent space to output layer, decode
    #         #     for iLayer in range(len(self.ADCNae) - 1, 0 - 1, -1):
    #         #         currnet = self.ADCNae[iLayer].network
    #         #         obj = currnet.train()
    #         #         obj = obj.to(self.device)
    #         #         zn_t = obj(zn_t, 2)
    #         #         recons_zn_feat = zn_t
    #         #
    #         # loss = Alpha * criterionDAL(recons_feat_t,recons_zn_feat)
    #         # # loss = Alpha * criterionDAL(latentAE_t,zn_t)

    #         # backward pass: compute gradient of the loss with respect to model parameters
    #         loss.backward()

    #         # perform a single optimization step (parameter update)
    #         optimizer.step()

    #     return x


    # ============================= Initialization =============================
    def initialization(self,dataStreamSource, winIdx, batchSize = 16, epoch = 50, device = torch.device('cpu')):

        # initialization phase: train CNN, train AE, train cluster, without clustering loss, evolving, epoch, add cluster in the end
        # always trained using labeled data
        # will create cluster for the last layer

        '''
        labeledDataSrc = dataStreamSource.labeledData;
        labeledDataTrg = dataStreamTarget.unlabeledData;
        '''
        labeledDataSrc=dataStreamSource.labeledData;
        #labeledDataTrg = dataStreamTarget.unlabeledData[0:200];

        labeledData=labeledDataSrc
        #labeledData = torch.cat((dataStreams.labeledData, dataStreamstwo.labeledData),dim=0)  # ini untuk sementara


        if winIdx == 0:
            self.labeledDataSrc  = labeledDataSrc
            #self.labeledDataTrg = labeledDataTrg
            self.unlabeledDataSrc = dataStreamSource.unlabeledData
            # self.unlabeledDataTrg = dataStreamTarget.unlabeledData
            #MNIST 70000

            self.batchSize    = batchSize

            self.device = device
            print('Network initialization phase is started')

            # Train CNN, for the first training, epoch is used
            extractedFeatures = self.trainCNN(labeledData, unlabeled = False, epoch = epoch)#initialization 50 epoch, data sudah gabungan target and source

            # Train AE, this trains the newly created layer
            latentFeatures = self.trainBasicAe(extractedFeatures, labeledData, evolving = self.evolving, winIdx = 0,
                                               epoch = epoch, unlabeled = False)
            # latentFeaturesSrc = self.trainBasicAeKL( dataStreamTarget.unlabeledData,  dataStreamTarget.unlabeledData, evolving = self.evolving, winIdx = 0, epoch = 1, unlabeled = True)
            # latentFeaturesTrg = self.trainBasicAeL1(extractedFeatures, self.unlabeledDataTrg, evolving=False, winIdx=0,
            #                                        epoch=epoch, unlabeled=False)
            # latentFeatures=torch.cat((latentFeaturesAE,latentFeaturesSrc,latentFeaturesTrg),dim=0)

            initialData    = latentFeatures.detach().clone().to('cpu')

            # create cluster, this is only done when there is no cluster for idx-th layer
            # self.createCluster(initialData, epoch = epoch)
            self.createCluster(initialData, epoch=epoch)

        if winIdx > 0:
            self.forwardADCN(labeledData, winIdx = winIdx)

            # Train AE, this trains the newly created layer
            latentFeatures = self.lastInputFeature          # latentFeatures is the extracted features from 2nd last layers
            latentFeatures = self.trainBasicAe(latentFeatures, labeledData, evolving = self.evolving, winIdx = winIdx, epoch = epoch, unlabeled = False)
            initialData    = latentFeatures.detach().clone().to('cpu')

            # create cluster, this is only done when there is no cluster for idx-th layer
            # pdb.set_trace()
            self.createCluster(initialData, epoch = epoch)

    def createCluster(self, initialData, epoch = 1):
        print('\n')
        print('Cluster initialization phase is started')

        nInput    = initialData.shape[1]

        myCluster = cluster(nInput, initialData[0:len(self.desiredLabels)],
                            nInitCluster = self.nInitCluster, clusterGrowing = self.clusterGrowing, desiredLabels = self.desiredLabels, device=self.device)

        # updateCluster
        myCluster.fit(initialData, epoch = epoch)
        self.clusterHistory = myCluster.clusterHistory
        print('A cluster was created containing ',myCluster.nCluster ,' centroids')

        # add cluster to the global cluster
        self.ADCNcluster = self.ADCNcluster + [myCluster]

    # ============================= Training =============================
    def trainCNN(self, x, unlabeled = True, epoch = 1):
        # x is image data with size 1 x 28 x 28
        nData = x.shape[0]
        x     = x.to(self.device)#x ini kalo dari initialization sudah gabungan target sama source

        # get optimizer
        optimizer = torch.optim.SGD(self.ADCNcnn.parameters(), lr = self.lr, momentum = 0.95, weight_decay = 0.00005)

        # prepare network
        self.ADCNcnn = self.ADCNcnn.train()
        self.ADCNcnn = self.ADCNcnn.to(self.device)

        # print('CNN training is started')
        for iEpoch in range(0, epoch): # ini diepoch 50 kali, kalo initialization

            # if iEpoch%20 == 0:
            #     print('Epoch: ', iEpoch)

            shuffled_indices = torch.randperm(nData)
            batchs = int(nData/self.batchSize)

            for i in range(batchs):

                # print('Data batch:',int(iData/self.batchSize))
                indices = shuffled_indices[i*self.batchSize:((i+1) * self.batchSize)]

                # load data
                minibatch_xTrain = x[indices]

                # clear the gradients of all optimized variables
                optimizer.zero_grad()

                # forward
                # latentAE here is the output of the deepest autoencoder
                latentFeatures = self.ADCNcnn(minibatch_xTrain)     # encode
                outputs        = self.ADCNcnn(latentFeatures, 2)    # decode

                # calculate the loss
                loss = self.criterion(outputs, minibatch_xTrain)

                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()

                # perform a single optimization step (parameter update)
                optimizer.step()


        with torch.no_grad():
            latentFeatures = self.ADCNcnn(x)

        # latentFeatures is the extracted features for the next hidden layer
        return latentFeatures.detach().clone()

    def trainBasicAe(self, x, x_ori, winIdx, epoch = 1, clustering = False, evolving = True, unlabeled = True):
        # grow layer: train CNN, train AE, train cluster
        # x is the extracted features from the previous hidden layer
        # this will update [winIdx]-th layer in greedy layer wise manner
        # TO DO: create grow and prune nodes, why batch size 1 is very bad, drift detection
        nData = x.shape[0]

        # masked input
        # maskedX = maskingNoise(x.detach().clone())              # make zero some of the input feature

        # prepare data
        x = x.to(self.device)
        # maskedX = maskedX.to(self.device)

        # prepare network
        self.ADCNae[winIdx].network = self.ADCNae[winIdx].network.train()
        self.ADCNae[winIdx].network = self.ADCNae[winIdx].network.to(self.device)

        self.growNode   = False
        self.pruneNode  = False
        evolve          = 0
        hiddenNodeHist  = []

        # print('AE training is started')
        for iEpoch in range(0, epoch):

            if iEpoch>0:
                evolving=False
            # print('=======Epoch==========TrainBasicAE:', iEpoch+1)

            shuffled_indices = torch.randperm(nData)
            batchs = int(nData/self.batchSize)

            for i in range(batchs):

                # print('Data batch:',int(iData/self.batchSize))
                indices = shuffled_indices[i*self.batchSize:((i+1) * self.batchSize)]

                # load data
                minibatch_xTrain = x[indices]         # input with masking noise
                minibatch_x      = x[indices]               # original input
                minibatch_x_ori  = x_ori[indices]

                if iEpoch == 0:
                    if self.batchSize > 1:
                        minibatch_x_mean = torch.mean(minibatch_x, dim=0).unsqueeze(dim=0)
                    else:
                        minibatch_x_mean = minibatch_x

                    # calculate mean of input
                    try:
                        self.averageInput[winIdx].updateMeanStd(minibatch_x_mean.detach().clone().to('cpu'))
                    except:
                        # if the number of input changes as a result of node evolution, the counter is reinitiated
                        self.averageInput[winIdx].reset()
                        self.averageInput[winIdx].updateMeanStd(minibatch_x_mean.detach().clone().to('cpu'))

                    # calculate bias and variance
                    bias, variance, HS = self.forwardBiasVar(self.averageInput[winIdx].mean, winIdx = winIdx)

                    # update bias and variance
                    self.updateBiasVariance(bias, variance, winIdx)
                else:
                    # calculate mean of input
                    bias, variance, HS = self.forwardBiasVar(self.averageInput[winIdx].mean, winIdx = winIdx)

                    # update bias and variance
                    self.updateBiasVariance(bias, variance, winIdx)

                if evolving:
                    # growing
                    self.growNodeIdentification(bias, winIdx)
                    if self.growNode:
                        self.hiddenNodeGrowing(winIdx)
                        # print('+++Grow node TrainBasicAE+++')

                        if clustering:
                            # add an input to the cluster
                            self.ADCNcluster[winIdx].growInput(HS[-1].item())

                    # pruning
                    self.pruneNodeIdentification(variance, winIdx)
                    if self.pruneNode:
                        # print('--- Prune node TrainBasicAE---')
                        self.findLeastSignificantNode(HS)
                        self.hiddenNodePruning(winIdx)
                        evolve = 1

                        if clustering:
                            # delete an input to the cluster
                            self.ADCNcluster[winIdx].deleteInput(self.leastSignificantNode)

                            # print(self.ADCNcluster[winIdx].centroids.shape)

                # clear the gradients of all optimized variables
                optimizer = torch.optim.SGD(self.ADCNae[winIdx].network.parameters(),
                                            lr = self.lr, momentum = 0.95, weight_decay = 0.00005)
                if clustering and self.clutering_loss:
                    optimizer.add_param_group({'params': [self.ADCNcluster[winIdx].centroids]})

                optimizer.zero_grad()

                # forward
                # latentAE here is the output of the deepest autoencoder
                latentAE = self.ADCNae[winIdx].network(minibatch_xTrain) # encoder
                outputs  = self.ADCNae[winIdx].network(latentAE, 2) # decoder

                ## calculate the loss
                # reconstruction loss
                rc_loss = self.criterion(outputs, minibatch_x)

                rc_loss.backward(retain_graph=True)
                optimizer.step()
                
                if clustering and not self.growNode and self.clutering_loss:
                    # kl loss
                    with torch.no_grad():
                        self.forwardADCN(minibatch_x_ori, winIdx)
                    pl = self.ADCNcluster[winIdx](self.latentFeatures)
                    ql = self.target_distribution(pl).detach()
                    
                    if not self.mmd:
                        kl = (torch.nn.KLDivLoss(size_average=False, reduction='mean')(F.log_softmax(pl,0), ql) / pl.shape[1])
                        kl_loss = self.alpha_kl * kl
                    else:
                        if self.kernel == 'gaussian':
                            mmd_loss = mmd_g(pl,ql)
                        elif self.kernel == 'multiscale':
                            mmd_loss = MMD(pl,ql,self.kernel,self.device)
                        elif self.kernel == 'rbf':
                            mmd_loss = MMD(pl,ql,self.kernel,self.device)
                        kl = mmd_loss                            # TODO: which kernel
                        kl_loss = self.alpha_kl * kl

                    # print(kl_loss)
                    if kl>0:
                        kl_loss.backward()
                        optimizer.step()

            self.growNode  = False
            self.pruneNode = False

            if evolving and self.growNode and clustering:
                # add an input to the cluster
                _, _, HS = self.forwardBiasVar(self.averageInput[winIdx].mean, winIdx = winIdx)
                self.ADCNcluster[winIdx].growInput(HS[-1].item())

            hiddenNodeHist.append(self.nHiddenNode)

        with torch.no_grad():
            latentAE = self.ADCNae[winIdx].network(x)

        if clustering:
            self.ADCNcluster[winIdx].fit(latentAE.detach().clone().to('cpu'), epoch)

        self.hiddenNodeHist = hiddenNodeHist

        # latentAE is the extracted features of winIdx-th AE for the next hidden layer
        return latentAE.detach().clone()

    def trainAE(self, x, x_ori, epoch = 1, clustering = True, evolving = True):
        # x is the extracted features from CNN encoder
        # this will update ALL layers of AE in greedy layer wise manner
        for idx, _ in enumerate(self.ADCNae):
            x = self.trainBasicAe(x, x_ori, idx, epoch = epoch, clustering = clustering, evolving = evolving)

    def trainDC(self, x, y, x1, x2):
        # encode decode in end-to-end manner

        # prepare network
        self.ADCNcnn = self.ADCNcnn.train()
        self.ADCNcnn = self.ADCNcnn.to(self.device)
        if self.domain_loss:
            self.discriminator = self.discriminator.train()
            self.discriminator = self.discriminator.to(self.device)

        # prepare data
        nData = x.shape[0]
        x = x.to(self.device)
        if self.domain_loss:
            y = y.to(self.device)

        # get optimizer
        optimizer = torch.optim.SGD(self.ADCNcnn.parameters(), lr=self.lr, momentum=0.95, weight_decay=0.00005)
        for iLayer, _ in enumerate(self.ADCNae):
            optimizer.add_param_group({'params': self.ADCNae[iLayer].network.parameters()})
        if self.domain_loss:
            optimizer.add_param_group({'params': self.discriminator.parameters()})

        # for loop of data batch
        shuffled_indices = torch.randperm(nData)
        batchs = int(nData/self.batchSize)

        for i in range(batchs):

            # print('Data batch:',int(iData/self.batchSize))
            indices = shuffled_indices[i*self.batchSize:((i+1) * self.batchSize)]
            
            minibatch_xTrain = x[indices]
            if self.domain_loss:
                domain_label = y[indices]
                # p = float(i + epoch * round(nData/self.batchSize)) / epoch / round(nData/self.batchSize)
                # alpha = 2. / (1. + np.exp(-10*p)) - 1

            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # forward encoder CNN
            feat_s = self.ADCNcnn(minibatch_xTrain)
            feat_s1 = feat_s

            # feedforward from input layer to latent space, encode
            for iLayer,_ in enumerate(self.ADCNae):
                currnet = self.ADCNae[iLayer].network
                obj     = currnet.train()
                obj     = obj.to(self.device)
                feat_s  = obj(feat_s)

            # feedforward from latent space to output layer, decode
            for iLayer in range(len(self.ADCNae)-1,0-1,-1):
                currnet = self.ADCNae[iLayer].network
                obj     = currnet.train()
                obj     = obj.to(self.device)
                feat_s  = obj(feat_s, 2)
                recons_feat_s = feat_s

            # # forward decoder CNN
            recons_x = self.ADCNcnn(feat_s, 2)

            # forward domain classifier
            domain_output = self.discriminator(feat_s1, 1.0)

            # recons loss cnn
            loss1 = self.criterion(recons_x, minibatch_xTrain) # clustering loss L1 in equ(14)
            # loss = loss1

            #recons loss AE
            loss2 = self.criterion(recons_feat_s, feat_s1)
            loss = loss1 + loss2

            # domain loss
            if self.domain_loss:
                loss3 = self.criterion(domain_output, domain_label)
                loss = loss1 + loss2 + self.alpha_dc * loss3
                # print(self.alpha_dc * loss3)

            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # perform a single optimization step (parameter update)
            optimizer.step()

        if self.mmd_d:
            dtype = torch.float

            # Initialize parameters
            epsilonOPT = torch.log(MatConvert(np.random.rand(1) * 10 ** (-10), self.device, dtype))
            epsilonOPT.requires_grad = True
            sigmaOPT = MatConvert(np.ones(1) * np.sqrt(2*300), self.device, dtype)
            sigmaOPT.requires_grad = True
            sigma0OPT = MatConvert(np.ones(1) * np.sqrt(0.005), self.device, dtype)
            sigma0OPT.requires_grad = True
            
            # for loop of data batch
            nData = min(x1.shape[0],x2.shape[0])
            shuffled_indices = torch.randperm(nData)
            batchs = int(nData/self.batchSize)

            # get optimizer
            list_param = list(self.ADCNcnn.parameters())
            for iLayer, _ in enumerate(self.ADCNae):
                list_param = list_param + list(self.ADCNae[iLayer].network.parameters())
            list_param = list_param + [epsilonOPT] + [sigmaOPT] + [sigma0OPT]
            optimizer = torch.optim.SGD(list_param, lr=self.lr, momentum=0.95, weight_decay=0.00005)

            for i in range(batchs):
                # print('Data batch:',int(iData/self.batchSize))
                indices = shuffled_indices[i*self.batchSize:((i+1) * self.batchSize)]
                
                minibatch_x1 = x1[indices].to(self.device)
                minibatch_x2 = x2[indices].to(self.device)

                # clear the gradients of all optimized variables
                optimizer.zero_grad()

                # forward encoder CNN
                feat_s1 = self.ADCNcnn(minibatch_x1)
                feat_s2 = self.ADCNcnn(minibatch_x2)

                # feedforward from input layer to latent space, encode
                for iLayer,_ in enumerate(self.ADCNae):
                    currnet = self.ADCNae[iLayer].network
                    obj     = currnet.train()
                    obj     = obj.to(self.device)
                    feat_s1  = obj(feat_s1)
                    feat_s2  = obj(feat_s2)

                # Compute epsilon, sigma and sigma_0
                ep = torch.exp(epsilonOPT) / (1 + torch.exp(epsilonOPT))
                sigma = sigmaOPT ** 2
                sigma0_u = sigma0OPT ** 2

                # mmd loss
                TEMP = MMDu(feat_s1, feat_s2, minibatch_x1, minibatch_x2, sigma, sigma0_u, ep)
                mmd_value_temp = -1 * (TEMP[0])
                mmd_std_temp = torch.sqrt(TEMP[1] + 10 ** (-8))
                STAT_u = torch.div(mmd_value_temp, mmd_std_temp) * self.alpha_dc
                print('mmd:',STAT_u)
                # Compute gradient
                STAT_u.backward()
                # Update weights using gradient descent
                optimizer.step()

        return x

    def trainMMD(self, x1, x2, x):
        # encode decode in end-to-end manner

        # prepare network
        self.ADCNcnn = self.ADCNcnn.train()
        self.ADCNcnn = self.ADCNcnn.to(self.device)

        # prepare data
        nData = x.shape[0]
        x = x.to(self.device)

        # get optimizer
        optimizer = torch.optim.SGD(self.ADCNcnn.parameters(), lr=self.lr, momentum=0.95, weight_decay=0.00005)
        for iLayer, _ in enumerate(self.ADCNae):
            optimizer.add_param_group({'params': self.ADCNae[iLayer].network.parameters()})

        # for loop of data batch
        shuffled_indices = torch.randperm(nData)
        batchs = int(nData/self.batchSize)

        for i in range(batchs):

            # print('Data batch:',int(iData/self.batchSize))
            indices = shuffled_indices[i*self.batchSize:((i+1) * self.batchSize)]
            
            minibatch_xTrain = x[indices]

            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # forward encoder CNN
            feat_s = self.ADCNcnn(minibatch_xTrain)
            feat_s1 = feat_s

            # feedforward from input layer to latent space, encode
            for iLayer,_ in enumerate(self.ADCNae):
                currnet = self.ADCNae[iLayer].network
                obj     = currnet.train()
                obj     = obj.to(self.device)
                feat_s  = obj(feat_s)

            # feedforward from latent space to output layer, decode
            for iLayer in range(len(self.ADCNae)-1,0-1,-1):
                currnet = self.ADCNae[iLayer].network
                obj     = currnet.train()
                obj     = obj.to(self.device)
                feat_s  = obj(feat_s, 2)
                recons_feat_s = feat_s

            # # forward decoder CNN
            recons_x = self.ADCNcnn(feat_s, 2)

            # recons loss cnn
            loss1 = self.criterion(recons_x, minibatch_xTrain) # clustering loss L1 in equ(14)
            # loss = loss1

            #recons loss AE
            loss2 = self.criterion(recons_feat_s, feat_s1)
            
            # mmd loss
            loss = loss1 + loss2

            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # perform a single optimization step (parameter update)
            optimizer.step()

        dtype = torch.float

        # Initialize parameters
        epsilonOPT = torch.log(MatConvert(np.random.rand(1) * 10 ** (-10), self.device, dtype))
        epsilonOPT.requires_grad = True
        sigmaOPT = MatConvert(np.ones(1) * np.sqrt(2*300), self.device, dtype)
        sigmaOPT.requires_grad = True
        sigma0OPT = MatConvert(np.ones(1) * np.sqrt(0.005), self.device, dtype)
        sigma0OPT.requires_grad = True
        
        # for loop of data batch
        nData = min(x1.shape[0],x2.shape[0])
        shuffled_indices = torch.randperm(nData)
        batchs = int(nData/self.batchSize)

        # get optimizer
        list_param = list(self.ADCNcnn.parameters())
        for iLayer, _ in enumerate(self.ADCNae):
            list_param = list_param + list(self.ADCNae[iLayer].network.parameters())
        list_param = list_param + [epsilonOPT] + [sigmaOPT] + [sigma0OPT]
        optimizer = torch.optim.SGD(list_param, lr=self.lr, momentum=0.95, weight_decay=0.00005)

        for i in range(batchs):
            # print('Data batch:',int(iData/self.batchSize))
            indices = shuffled_indices[i*self.batchSize:((i+1) * self.batchSize)]
            
            minibatch_x1 = x1[indices].to(self.device)
            minibatch_x2 = x2[indices].to(self.device)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # forward encoder CNN
            feat_s1 = self.ADCNcnn(minibatch_x1)
            feat_s2 = self.ADCNcnn(minibatch_x2)

            # feedforward from input layer to latent space, encode
            for iLayer,_ in enumerate(self.ADCNae):
                currnet = self.ADCNae[iLayer].network
                obj     = currnet.train()
                obj     = obj.to(self.device)
                feat_s1  = obj(feat_s1)
                feat_s2  = obj(feat_s2)

            # Compute epsilon, sigma and sigma_0
            ep = torch.exp(epsilonOPT) / (1 + torch.exp(epsilonOPT))
            sigma = sigmaOPT ** 2
            sigma0_u = sigma0OPT ** 2

            # mmd loss
            TEMP = MMDu(feat_s1, feat_s2, minibatch_x1, minibatch_x2, sigma, sigma0_u, ep)
            mmd_value_temp = -1 * (TEMP[0])
            mmd_std_temp = torch.sqrt(TEMP[1] + 10 ** (-8))
            STAT_u = torch.div(mmd_value_temp, mmd_std_temp)
            print('mmd:',STAT_u)
            # Compute gradient
            STAT_u.backward()
            # Update weights using gradient descent
            optimizer.step()

        return x

    # def trainDC_one(self, x, y):
    #     # encode decode in end-to-end manner

    #     # prepare data
    #     shuffled_indices = torch.randperm(np.shape(x)[0])
    #     img_x = x[shuffled_indices]
    #     x  =  x[shuffled_indices]
    #     if self.domain_loss:
    #         domain_label = y[shuffled_indices]

    #     # prepare network
    #     self.ADCNcnn = self.ADCNcnn.train()
    #     self.ADCNcnn = self.ADCNcnn.to(self.device)
    #     if self.domain_loss:
    #         self.discriminator = self.discriminator.train()
    #         self.discriminator = self.discriminator.to(device)

    #     # get optimizer
    #     optimizer = torch.optim.SGD(self.ADCNcnn.parameters(), lr=self.lr, momentum=0.95, weight_decay=0.00005)
    #     for iLayer, _ in enumerate(self.ADCNae):
    #         optimizer.add_param_group({'params': self.ADCNae[iLayer].network.parameters()})
    #     if self.domain_loss:
    #         optimizer.add_param_group({'params': self.discriminator.parameters()})

    #     # clear the gradients of all optimized variables
    #     optimizer.zero_grad()

    #     # forward encoder CNN
    #     x = self.ADCNcnn(x)
    #     feat_s = x

    #     # feedforward from input layer to latent space, encode
    #     for iLayer,_ in enumerate(self.ADCNae):
    #         currnet = self.ADCNae[iLayer].network
    #         obj     = currnet.train()
    #         obj     = obj.to(self.device)
    #         x  = obj(x)

    #     # feedforward from latent space to output layer, decode
    #     for iLayer in range(len(self.ADCNae)-1,0-1,-1):
    #         currnet = self.ADCNae[iLayer].network
    #         obj     = currnet.train()
    #         obj     = obj.to(self.device)
    #         x       = obj(x, 2)
    #         recons_feat_s = x

    #     # # forward decoder CNN
    #     recons_x = self.ADCNcnn(x, 2)

    #     # forward domain classifier
    #     domain_output = self.discriminator(feat_s, 1.0)

    #     # recons loss cnn
    #     loss1 = self.criterion(recons_x, img_x) # clustering loss L1 in equ(14)
    #     # loss = loss1

    #     #recons loss AE
    #     loss2 = self.criterion(recons_feat_s, feat_s)
    #     loss = loss1 + loss2

    #     # domain loss
    #     if self.domain_loss:
    #         loss3 = self.criterion(domain_output, domain_label)
    #         loss = loss1 + loss2 + self.alpha_dc * loss3
    #         print(loss3)

    #     # backward pass: compute gradient of the loss with respect to model parameters
    #     loss.backward()

    #     # perform a single optimization step (parameter update)
    #     optimizer.step()

    #     return x


    def train_main(self, batchDataS, batchDataT, nEpochPreq=1, nEpochKLL1=1):
        # train ADCN, all network structures, layer wise manner
        x = torch.cat((batchDataS,batchDataT),dim=0)
        extractedFeatures = self.trainCNN(x, unlabeled = True, epoch = nEpochPreq)
        self.trainAE(extractedFeatures, x, epoch = nEpochPreq, evolving = self.evolving)

        # domain label (0-source data 1-target data)
        if self.domain_loss:
            sourcelabel = torch.zeros(batchDataS.shape[0])
            targetlabel = torch.ones(batchDataT.shape[0])
            domain_label = torch.cat((sourcelabel, targetlabel),0)
            domain_label = domain_label.long()
            domain_label = torch.zeros(x.shape[0], 2).scatter_(1, domain_label.unsqueeze(1), 1)
        else:
            domain_label = None
    
        for iEpoch in range(0, nEpochKLL1):
            # print('=======Epoch==========TrainDomain:', iEpoch+1)
            self.trainDC(x, domain_label, batchDataS, batchDataT)

            # if not self.mmd_d:
            #     self.trainDC(x, domain_label)
            # else:
            #     self.trainMMD(batchDataS, batchDataT, x)

            # self.trainDC_one(x, domain_label)


    # def fitLeo(self, batchDataS, batchDataT, nEpochPreq = 1, nEpochKLL1=1):

    #     # train ADCN, all network structures, layer wise manner
    #     # x is image data with size 1 x 28 x 28
    #     x = torch.cat((batchDataS,batchDataT),dim=0)
    #     # self.optimizeCNN(x) # there is a random function already to shuffle batchDataS and batchDataT
    #     extractedFeatures = self.trainCNN(x, unlabeled = True, epoch = nEpochPreq)#initialization 50 epoch, data sudah gabungan target and source
    #     self.trainAE(extractedFeatures, x, epoch = nEpochPreq, evolving = self.evolving)
    #     # recons loss cnn

    #     for i in range(0,nEpochKLL1):
    #         print("epoch for KL and L1 :",i+1, "  from ",nEpochKLL1)
    #         self.optimizeSource(batchDataS, i, nEpochKLL1)
    #         self.optimizeTarget(batchDataT, i, nEpochKLL1)


    #     # self.fitLeopard(batchDataS, batchDataT, winIdx=0,epoch=epoch, clustering=True, evolving=True)
    #     # self.trainBasicAeL1(extractedFeatures,batchDataT, winIdx=0,epoch=1, evolving=False)

    def updateAllegiance(self, labeledData, labeledLabel, randomTesting = False):
        # At the end of each phase, we introduce a limited amount of labeled data per class to evaluate classification accuracy.
        # forward to each layer
        for iLayer in range(len(self.ADCNae)):
            self.forwardADCN(labeledData, winIdx = iLayer)

            # update allegiance in each layer
            self.ADCNcluster[iLayer].updateAllegiance(self.latentFeatures.detach().clone().to('cpu'),
                                                      labeledLabel.to('cpu'), randomTesting = randomTesting)
            # self.ADCNcluster[iLayer].updateAllegiance(self.latentFeatures.detach().clone().to('cpu').numpy(),
            #                                           labeledLabel.to('cpu').numpy(), randomTesting = randomTesting)

    # ============================= Testing =============================
    def predict(self, x):
    # def predict(self, x, y): # revise
        with torch.no_grad():
            x = x.to(self.device)

            # prepare network
            self.ADCNcnn = self.ADCNcnn.eval()
            self.ADCNcnn = self.ADCNcnn.to(self.device)

            # forward encoder CNN
            x = self.ADCNcnn(x)

            # feedforward from input layer to latent space
            score = np.zeros([x.shape[0], self.nOutput])
            for iLayer in range(len(self.ADCNae)):
                currnet = self.ADCNae[iLayer].network
                obj     = currnet.eval()
                obj     = obj.to(self.device)
                x       = obj(x)

                # predict
                self.ADCNcluster[iLayer].predict(x.detach().clone().to('cpu'))
                # self.ADCNcluster[iLayer].predict(x.detach().clone().to('cpu'),y.detach().clone().to('cpu')) # revised
                # pdb.set_trace()
                score = score + stableSoftmax(self.ADCNcluster[iLayer].score.numpy())

        self.predictedLabel = np.argmax(score,1)
        self.score          = score

    def testing(self, x, label):
        # testing
        start_test          = time.time()
        self.predict(x)
        # self.predict(x,label) # revised
        end_test            = time.time()
        self.testingTime    = end_test - start_test
        correct             = (self.predictedLabel == label.numpy()).sum().item()
        self.accuracy       = 100*correct/(self.predictedLabel == label.numpy()).shape[0]  # 1: correct, 0: wrong
        self.trueClassLabel = label.numpy()

    # def randomTesting(self, x, label):
    #     # testing
    #     start_test          = time.time()
    #     self.predict(x)
    #     end_test            = time.time()
    #     self.testingTime    = end_test - start_test
    #     randomPrediction    = np.random.randint(self.nOutput, size=self.predictedLabel.shape[0])
    #     correct             = (randomPrediction == label.numpy()).sum().item()
    #     self.accuracy       = 100*correct/(self.predictedLabel == label.numpy()).shape[0]  # 1: correct, 0: wrong
    #     self.trueClassLabel = label.numpy()

    # ============================= Evolving mechanism =============================
    def layerGrowing(self):
        if len(self.ADCNae) == len(self.ADCNcluster) and self.ADCNae[-1].nNodes > 4 :
        # if len(self.ADCNae) == len(self.ADCNcluster):
            self.ADCNae       = self.ADCNae + [smallAE(self.ADCNae[-1].nNodes,int(self.ADCNae[-1].nNodes/2))]
            self.averageBias  = self.averageBias  + [meanStdCalculator()]
            self.averageVar   = self.averageVar   + [meanStdCalculator()]
            self.averageInput = self.averageInput + [meanStdCalculator()]
            self.nHiddenLayer = len(self.ADCNae)
            print('*** ADD a new LAYER ***')

    def hiddenNodeGrowing(self, winIdx):
        if winIdx <= (len(self.ADCNae)-1):
            copyNet = copy.deepcopy(self.ADCNae[winIdx])
            copyNet.nodeGrowing(device = self.device)
            self.ADCNae[winIdx] = copy.deepcopy(copyNet)
            if winIdx != (len(self.ADCNae)-1):
                copyNextNet = copy.deepcopy(self.ADCNae[winIdx+1])
                copyNextNet.inputGrowing(device = self.device)
                self.ADCNae[winIdx+1] = copy.deepcopy(copyNextNet)

            # print('+++ GROW a hidden NODE +++')
            self.updateNetProperties()
        else:
            raise IndexError

    def hiddenNodePruning(self, winIdx):
        if winIdx <= (len(self.ADCNae)-1):
            copyNet = copy.deepcopy(self.ADCNae[winIdx])
            copyNet.nodePruning(self.leastSignificantNode)
            self.ADCNae[winIdx] = copy.deepcopy(copyNet)
            if winIdx != (len(self.ADCNae)-1):
                copyNextNet = copy.deepcopy(self.ADCNae[winIdx+1])
                copyNextNet.inputPruning(self.leastSignificantNode)
                self.ADCNae[winIdx+1] = copy.deepcopy(copyNextNet)

            # print('--- Hidden NODE No: ',self.leastSignificantNode,' is PRUNED ---')
            self.updateNetProperties()
        else:
            raise IndexError

    # ============================= Network Evaluation =============================
    # def LwFloss(self, currentBatchOutput, currentBatchData):
    #     loss = []
    #     criterion = nn.BCELoss()
    #     for iTask,_ in enumerate(self.ADCNold):
    #         with torch.no_grad():
    #             minibatch_xOld = self.ADCNold[iTask].forward(currentBatchData, self.device)     # it acts as the target
    #
    #         regStr = self.regStrLWF*(1.0 - self.nOutputPerTask/((iTask + 1)*self.nOutputPerTask + self.nOutputPerTask))
    #         loss.append(regStr*criterion(currentBatchOutput, minibatch_xOld.detach().clone()))
    #
    #     return sum(loss)

    def driftDetection(self, batchData, prevBatchData = None):
        self.ADCNcnn = self.ADCNcnn.to(self.device)
        batchData    = batchData.to(self.device)

        with torch.no_grad():
            # forward encoder CNN
            currFeatureMatrix = self.ADCNcnn(batchData)
            currFeatureMatrix = currFeatureMatrix.to('cpu')

            if prevBatchData is not None:
                prevBatchData = prevBatchData.to(self.device)

                # forward encoder CNN
                prevFeatureMatrix = self.ADCNcnn(prevBatchData)
                prevFeatureMatrix = prevFeatureMatrix.to('cpu')
            # currFeatureMatrix is a list containing the mean of extracted features

        self.driftStatusOld = self.driftStatus
        driftStatus = 0  # 0: no drift, 1: warning, 2: drift

        if self.driftStatusOld != 2:
            # detect drift
            # combine buffer data, when previous batch is warning
            if self.driftStatusOld == 1:
                with torch.no_grad():
                    bufferFetureMatrix = self.bufferFetureMatrix.to(self.device)

                    # forward encoder CNN
                    bufferFetureMatrix = self.ADCNcnn(bufferFetureMatrix)
                    bufferFetureMatrix = bufferFetureMatrix.to('cpu')

                    currFeatureMatrix = torch.cat((bufferFetureMatrix,currFeatureMatrix),0)

            # combine current and previous feature matrix
            combinedFeatureMatrix = currFeatureMatrix
            if prevBatchData is not None:
                combinedFeatureMatrix = torch.cat((prevFeatureMatrix,currFeatureMatrix),0)

            # prepare statistical coefficient to confirm a cut point
            nData             = combinedFeatureMatrix.shape[0]
            cutPointCandidate = [int(nData/4),int(nData/2),int(nData*3/4)]
            cutPoint          = 0
            miu_X             = torch.mean(combinedFeatureMatrix,0)    # statistics of the current batch data extracted features
            errorBoundX       = np.sqrt((1/(2*nData))*np.log(1/self.alphaDrift))

            # confirm the cut point
            for iCut in cutPointCandidate:
                miu_V       = torch.mean(combinedFeatureMatrix[0:iCut], 0)
                nV          = combinedFeatureMatrix[0:iCut].shape[0]
                errorBoundV = np.sqrt((1/(2*nV))*np.log(1/self.alphaDrift))
                if torch.mean(miu_X + errorBoundX).item() <= torch.mean(miu_V + errorBoundV).item():
                    cutPoint = iCut
                    print('A cut point is detected cut: ', cutPoint)
                    break

            # confirm drift
            # if np.abs(miu_F - miu_E) >= errorBoundDrift:   # This formula is able to detect drift, even the performance improves
            if cutPoint > 0:
                # prepare statistical coefficient to confirm a drift
                max_b,_ = torch.max(combinedFeatureMatrix, 0)
                min_a,_ = torch.min(combinedFeatureMatrix, 0)
                diff_ba = max_b - min_a

                errorBoundDrift = torch.mean(diff_ba*np.sqrt(((nData - nV)/(2*nV*nData))*np.log(1/self.alphaDrift))).item()
                # if miu_V - miu_X >= errorBoundDrift:   # This formula is only able to detect drift when the performance decreses
                # print('np.abs(miu_V - miu_X) : ', np.abs(miu_V - miu_X),'>','errorBoundDrift',errorBoundDrift)

                if torch.mean(np.abs(miu_V - miu_X)).item() >= errorBoundDrift:   # This formula is able to detect drift, even the performance improves
                    # confirm drift
                    print('H0 is rejected with size: ', errorBoundDrift)
                    print('Status: DRIFT')
                    driftStatus             = 2
                    self.bufferFetureMatrix = []
                    # self.prevFeatureMatrix  = []

                else:
                    # prepare statistical coefficient to confirm a warning
                    errorBoundWarning = torch.mean(diff_ba*np.sqrt(((nData - nV)/(2*nV*nData))*np.log(1/self.alphaWarning))).item()
                    # print('np.abs(miu_V - miu_X) : ', np.abs(miu_V - miu_X),'>','errorBoundWarning',errorBoundWarning)
                    # if there is a warning in the previous batch, then there is only 2 option left: drift or stable.
                    # it is assumed that the number of samples is large enough to confirm those 2 situation
                    # if miu_V - miu_X >= errorBoundWarning and self.driftStatusOld != 1:
                    if torch.mean(np.abs(miu_V - miu_X)).item() >= errorBoundWarning and self.driftStatusOld != 1:
                        # confirm warning
                        # if there is a warning, the currFeatureMatrix is stored in the buffer and will be evaluated in the next batch
                        # together with the currFeatureMatrix of the next batch
                        print('H0 is rejected with size: ', errorBoundWarning)
                        print('Status: WARNING')
                        driftStatus = 1
                        self.bufferFetureMatrix = prevBatchData.to('cpu')
                    else:
                        # confirm stable
                        print('H0 is NOT rejected, size:', torch.mean(np.abs(miu_V - miu_X)).item(),
                              '; Error bound warning: ', errorBoundWarning, '; Error bound drift: ', errorBoundDrift)
                        print('Status: STABLE')
                        driftStatus = 0
                        self.bufferFetureMatrix = []
                        # self.prevFeatureMatrix  = currFeatureMatrix
            else:
                # there is no cutpoint detected means that there is no significant increase in the combinedFeatureMatrix
                print('Cut point is NOT detected')
                print('Status: STABLE')
                driftStatus = 0
                # self.prevFeatureMatrix = currFeatureMatrix

        self.driftStatus = driftStatus
        self.driftHistory.append(driftStatus)

    def growNodeIdentification(self, bias, winIdx):
        # confirm high variance situation
        # winIdx is the indes of current AE network, started from 0 (np.log2(winIdx) + 1)
        winIdxa           = winIdx + 1       # winIdx start from 0, that is why it is required to add 1
        # dynamicKsigmaGrow = (np.log(winIdxa) + 1)*(1.3*np.exp(-bias) + 0.7)
        dynamicKsigmaGrow = (1.3*np.exp(-bias) + 0.7)
        growCondition1    = (self.averageBias[winIdx].minMean +
                             dynamicKsigmaGrow*self.averageBias[winIdx].minStd)
        growCondition2    = self.averageBias[winIdx].mean + self.averageBias[winIdx].std

        # print('growCondition2: ', growCondition2,'growCondition1: ', growCondition1)
        if growCondition2 > growCondition1 and self.averageBias[winIdx].count >= 1:
            self.growNode = True
            # if winIdx > 0:
            #     pdb.set_trace()
        else:
            self.growNode = False

    def pruneNodeIdentification(self, variance, winIdx):
        # confirm high variance situation
        # winIdx is the indes of current AE network, started from 0
        winIdxa            = winIdx + 1      # winIdx start from 0, that is why it is required to add 1
        # dynamicKsigmaPrune = (np.log(winIdxa) + 1)*(1.3*np.exp(-variance) + 0.7)
        dynamicKsigmaPrune = (1.3*np.exp(-variance) + 0.7)
        pruneCondition1    = (self.averageVar[winIdx].minMean +
                              2*dynamicKsigmaPrune*self.averageVar[winIdx].minStd)
        pruneCondition2    = self.averageVar[winIdx].mean + self.averageVar[winIdx].std

        if (pruneCondition2 > pruneCondition1 and not self.growNode and
                self.averageVar[winIdx].count >= 20 and
                self.ADCNae[winIdx].nNodes > self.nOutput):
            self.pruneNode = True
        else:
            self.pruneNode = False

    def findLeastSignificantNode(self, hiddenNodeSignificance):
        # find the least significant node given the hidden node significance in the current layer, only for AE network
        # should be executed after doing feedforwardBiasVar on the winning layer
        self.leastSignificantNode = torch.argmin(torch.abs(hiddenNodeSignificance)).tolist()
        # print('Pruned node: ', self.leastSignificantNode)

    def updateBiasVariance(self, bias, variance, winIdx):
        # calculate mean of bias and variance
        # should be executed after doing feedforwardBiasVar on the winning layer
        self.averageBias[winIdx].updateMeanStd(bias)
        if self.averageBias[winIdx].count < 1 or self.growNode:
            self.averageBias[winIdx].resetMinMeanStd()
            # if winIdx > 0:
            #     pdb.set_trace()
        else:
            self.averageBias[winIdx].updateMeanStdMin()

        # calculate mean of variance
        self.averageVar[winIdx].updateMeanStd(variance)
        if self.averageVar[winIdx].count < 20 or self.pruneNode:
            self.averageVar[winIdx].resetMinMeanStd()
        else:
            self.averageVar[winIdx].updateMeanStdMin()