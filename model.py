import numpy as np
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
# from numpy import linalg as LA
import pdb
import random
from utilsADCN import deleteRowTensor, deleteColTensor, ReverseLayerF


# random seed control
# np.random.seed(0)
# torch.manual_seed(0)
# random.seed(0)

# ============================= Cluster model =============================
class cluster(nn.Module):
    def __init__(self, nInput, initData=None, nInitCluster=2, clusterGrowing=True, desiredLabels=[0, 1], device='cpu'):
        super(cluster, self).__init__()
        # create centroids
        self.nInput = nInput
        self.nClass = len(desiredLabels)
        self.nCluster = nInitCluster

        self.device = device

        if initData is None or nInitCluster > 2:
            self.centroids = nn.Parameter(torch.rand(self.nCluster, nInput))
            nn.init.xavier_uniform_(self.centroids)
        else:
            self.centroids = nn.Parameter(initData)
            self.nCluster = self.centroids.shape[0]

        self.support = torch.ones(self.nCluster, dtype=int)

        # initial distance
        self.meanDistance = torch.ones(self.nCluster)
        self.stdDistance = torch.zeros(self.nCluster)

        self.growClustInd = False
        self.clusterGrowing = clusterGrowing
        self.insignificantCluster = []

        self.alpha = 1.0

    # ============================= evaluation =============================
    def distanceMeasure(self, data):
        self.data = data
        l2NormDistance = torch.norm(self.centroids.detach() - data, dim=1)
        self.nearestCentroid = np.argmin(l2NormDistance.detach().numpy(), 0)

    def detectNovelPattern(self, newData, winCentroid, winClsIdx):
        r'''

        Args:
            newData:        training batch feature
            winCentroid:    winning cluster center
            winClsIdx:      winning cluster index

        detectNovelPattern corresponds to evolution of cluster of IV-C Structural Learning of LEOPARD.

        '''

        winDistance = torch.norm(winCentroid - newData, dim=1)

        # estimate distance
        miuOld = self.meanDistance[winClsIdx]
        self.meanDistance[winClsIdx] = (self.meanDistance[winClsIdx] - ((self.meanDistance[winClsIdx] - winDistance) /
                                                                        (self.support[winClsIdx])))
        variance = ((self.stdDistance[winClsIdx]) ** 2 + (miuOld) ** 2 -
                    (self.meanDistance[winClsIdx]) ** 2 + (((winDistance) ** 2 -
                                                            (self.stdDistance[winClsIdx]) ** 2 - (miuOld) ** 2) /
                                                           (self.support[winClsIdx])))

        self.stdDistance[winClsIdx] = torch.sqrt(torch.abs(variance.detach())) # revised

        # grow cluster condition
        dynamicSigma = 2 * np.exp(-winDistance.detach().numpy()) + 2  # 2
        growClusterCondition = (self.meanDistance[winClsIdx] + torch.from_numpy(dynamicSigma) * self.stdDistance[winClsIdx])

        if winDistance > growClusterCondition:
            self.growClustInd = True
        else:
            self.growClustInd = False

    def identifyInsignificantCluster(self):
        self.insignificantCluster = []
        if self.nCluster > self.nClass:
            self.insignificantCluster = np.where(np.max(self.allegianceClusterClass, 1) < 0.55)[0].tolist()

    @staticmethod
    def target_distribution(x):
        weight = (x ** 2) / torch.sum(x, 0)
        return (weight.t() / torch.sum(weight,1)).t()

    # ============================= evolving =============================
    def growCluster(self, newCentroid):
        self.nCluster += 1
        self.centroids.data = torch.cat((self.centroids.data, newCentroid), 0)
        # del self.centroids.grad
        if self.centroids.grad is not None:
            self.centroids.grad = torch.cat((self.centroids.grad, torch.zeros_like(newCentroid)), 0)
        self.meanDistance = torch.cat((self.meanDistance, torch.unsqueeze(self.distance.min(),dim=0)), 0)
        self.stdDistance = torch.cat((self.stdDistance, torch.zeros(1)), 0)
        self.support = torch.cat((self.support, torch.ones(1).type(torch.int64)), 0)

    def removeCluster(self, index):
        self.nCluster -= len(index)
        self.centroids = np.delete(self.centroids, index, 0)
        # del self.centroids.grad
        if self.centroids.grad is not None:
            self.centroids.grad = (self.centroids.grad[:,torch.arange(self.centroids.grad.size(0))!=index])
        self.allegianceClusterClass = np.delete(self.allegianceClusterClass, index, 0)
        self.meanDistance = np.delete(self.meanDistance, index, 0)
        self.stdDistance = np.delete(self.stdDistance, index, 0)
        self.support = np.delete(self.support, index, 0)
        self.insignificantCluster = []

    def growInput(self, constant=None):
        self.nInput += 1
        if constant is None:
            # self.centroids.data = torch.cat((self.centroids.data, torch.zeros([self.nCluster, 1])), 1)
            # # del self.centroids.grad
            # if self.centroids.grad is not None:
            #     self.centroids.grad = torch.cat((self.centroids.grad, torch.zeros([self.nCluster, 1])), 1)
            with torch.no_grad():
                self.centroids = nn.Parameter(torch.cat((self.centroids, torch.ones([self.nCluster, 1])), 1))

        elif constant is not None:
            # self.centroids.data = torch.cat((self.centroids.data, constant * torch.ones([self.nCluster, 1])), 1)
            # # del self.centroids.grad
            # if self.centroids.grad is not None:
            #     self.centroids.grad = torch.cat((self.centroids.grad, torch.zeros([self.nCluster, 1])), 1)
            with torch.no_grad():
                self.centroids = nn.Parameter(torch.cat((self.centroids, constant * torch.ones([self.nCluster, 1])), 1))


    def deleteInput(self, index):
        self.nInput -= 1
        # self.centroids.data = (self.centroids.data[:,torch.arange(self.centroids.size(1))!=index]) # revised
        # # del self.centroids.grad
        # if self.centroids.grad is not None:
        #     self.centroids.grad = (self.centroids.grad[:,torch.arange(self.centroids.grad.size(1))!=index]) # revised
        with torch.no_grad():
            self.centroids = nn.Parameter(self.centroids[:,torch.arange(self.centroids.size(1))!=index])


    # ============================= update =============================
    def augmentLabeledSamples(self, data, labels):
        # data is the encoded data of the respective layer
        # data and labels are numpy arrays
        self.labeledData = torch.cat((self.labeledData, data), 0)
        self.labels = torch.cat((self.labels, labels), 0)

        # update labels count
        uniqueLabels = np.unique(labels)
        for iLabel in uniqueLabels:
            iLabelsCount = np.array([(data[labels == iLabel]).shape[0]])
            self.labelsCount[iLabel] = self.labelsCount[iLabel] + iLabelsCount

    def updateCentroid(self, newData, winClsIdx, addSupport=True, init=False):
        if addSupport:
            self.support[winClsIdx] = self.support[winClsIdx] + 1
        if init:
            with torch.no_grad():
                self.centroids[winClsIdx] = (self.centroids[winClsIdx] -
                                            ((self.centroids[winClsIdx] - newData) / (self.support[winClsIdx])))
        self.distance = torch.norm(self.centroids.detach() - newData, dim=1)

    def updateAllegiance(self, data, labels, randomTesting=False):
        # data and labels are in numpy array format

        # nData = data.shape[0]
        #
        # allegiance = torch.zeros(self.nCluster, nData)
        # centroids = self.centroids.detach()
        # for i, iCentroid in enumerate(centroids):
        #     allegiance[i] = (torch.exp(-torch.norm(iCentroid - data, dim=1)))
        # self.allegiance = allegiance / torch.max(allegiance, 0)[0]  # equation 6 from paper STEM

        xe = torch.unsqueeze(data,1) - self.centroids.detach()
        q = 1.0 / (1.0 + (torch.sum(torch.mul(xe,xe), 2) / self.alpha))
        q = q ** (self.alpha + 1.0) / 2.0
        self.allegiance = (q.t() / torch.sum(q, 1)) # DEC

        uniqueLabels = np.unique(labels)

        allegianceClusterClass = torch.zeros(self.nCluster, self.nClass)
        for iCluster in range(0, self.nCluster):
            for iClass in uniqueLabels:
                # allegianceClusterClass[iCluster][iClass] = (torch.sum(self.allegiance[iCluster][labels == iClass]) /
                #                                             len(labels[labels == iClass]))  # equation 7 from paper STEM
                allegianceClusterClass[iCluster][iClass] = (torch.sum(self.allegiance[iCluster][labels == iClass]) /
                                                            torch.sum(self.allegiance[iCluster]))  # equation 7 from paper STEM

        # pdb.set_trace()
        self.allegianceClusterClass = F.softmax(allegianceClusterClass, dim=1)

    def forward(self, data):
        # student t-distribution, as same as used in t-SNE algorithm.
        # q_ij = 1/(1+dist(x_i,u_j)^2), then normalize it.
        # data and labels are in numpy array format

        # nData = data.shape[0]
        # # allegiance = np.empty([self.nCluster,nData],dtype=float)*0.0
        # allegiance = torch.zeros(self.nCluster, nData)
        # for i in range(self.nCluster):
        #     allegiance[i] = (torch.exp(-torch.norm(self.centroids[i] - data, dim=1)))
        # phi_max, _ = torch.max(allegiance, dim=0)
        # q_ = allegiance / phi_max  # equation 6 from paper STEM

        # center = centroidupdate.apply(self.centroids)
        xe = torch.unsqueeze(data.to(self.device),1) - self.centroids.to(self.device)
        q = 1.0 / (1.0 + (torch.sum(torch.mul(xe,xe), 2) / self.alpha))
        q = q ** (self.alpha + 1.0) / 2.0
        q_ = (q.t() / torch.sum(q, 1)) # DEC

        return q_

    def fit(self, trainData, epoch=1, init=False):
        nTrainData = trainData.shape[0]

        growCount = 0

        clusterHistory = []

        for iEpoch in range(0, epoch):
            shuffled_indices = torch.randperm(nTrainData)

            for iData in range(0, nTrainData):
                # load data
                indices = shuffled_indices[iData:iData + 1]
                minibatch_xTrain = trainData[indices]
                # minibatch_xTrain = minibatch_xTrain

                self.distanceMeasure(minibatch_xTrain)

                # print(self.nearestCentroid)

                # update clusters
                if iEpoch == 0:
                    self.updateCentroid(minibatch_xTrain, self.nearestCentroid, init=init)
                else:
                    self.updateCentroid(minibatch_xTrain, self.nearestCentroid, addSupport=False, init=init)

                self.detectNovelPattern(minibatch_xTrain, self.centroids[self.nearestCentroid].detach(), self.nearestCentroid)

                if self.growClustInd and self.clusterGrowing:
                    # grow clusters
                    self.growCluster(minibatch_xTrain)
                    growCount += 1

            clusterHistory.append(self.nCluster)

        # reassigning empty clusters
        self.clusterReassigning()

        self.clusterHistory = clusterHistory
        # update allegiance
        # self.updateAllegiance()

    def clusterReassigning(self):
        if self.support.min() == 1:
            singleToneCluster = np.where(self.support <= 2)[0].tolist()
            randCandidateList = np.where(self.support > 50)[0]

            if len(singleToneCluster) > 0 and len(randCandidateList.tolist()) > 1:  # revised
                with torch.no_grad():
                    for iClusterIdx in singleToneCluster:
                        # randCandidateIdx  = random.choice(randCandidateList)
                        randCandidateIdx = randCandidateList[torch.randperm(len(randCandidateList))][0]
                        self.centroids[iClusterIdx] = self.centroids[randCandidateIdx] + 0.007  # np.random.rand(1)

    # ============================= prediction =============================
    def predict(self, testData):
        nTestData = testData.shape[0]
        # testData = testData.numpy()
        prediction = torch.zeros([nTestData, self.nClass])
        for i, iAllegiance in enumerate(self.allegianceClusterClass):
            # if torch.max(iAllegiance) > (1 / self.nClass + 0.0001):
            #     centroidAllegiance = (iAllegiance * torch.unsqueeze(np.exp(
            #         -torch.norm(testData - self.centroids[i].detach(), dim=1)), 0).T)
            #     prediction += centroidAllegiance
            if torch.max(iAllegiance) > (1 / self.nClass + 0.0001):
                xe = testData.unsqueeze(1) - self.centroids[i].detach()
                q = 1.0 / (1.0 + (torch.sum(torch.mul(xe,xe), 2) / self.alpha))
                q = q ** (self.alpha + 1.0) / 2.0
                centroidAllegiance = (iAllegiance * torch.unsqueeze(q, 0).T)
                prediction += centroidAllegiance.squeeze()

            # # varBar = np.round(LA.norm(self.centroids[i] - testData, axis=1), 2)
            # centroidAllegiance = (iAllegiance*np.expand_dims(np.exp(
            #     -LA.norm(self.centroids[i] - testData,axis=1)),0).T)
            # # equation 8 from paper STEM
            # # centroidAllegiance = (iAllegiance * np.expand_dims(np.exp(
            # #     -varBar), 0).T)
            #
            # prediction += centroidAllegiance                                # equation 8 from paper STEM

        self.score = prediction
        self.predictedLabels = np.argmax(prediction, 1)  # equation 8 from paper STEM

    def getCluster(self, testData):
        nTestData = testData.shape[0]
        # testData = testData.numpy()
        score = np.zeros((self.nCluster, nTestData), dtype=float).astype(np.float32)
        # score      = np.empty([self.nCluster,nTestData],dtype=float)*0.0

        for i, iCentroid in enumerate(self.centroids):
            distance = torch.norm(iCentroid - testData, dim=1)
            score[i] = distance

        self.predictedClusters = np.argmax(score, 0)

# ============================= Main network =============================
class smallAE():
    def __init__(self, no_input, no_hidden):
        self.network = basicAE(no_input, no_hidden)
        print(no_input, no_hidden)
        self.netUpdateProperties()

    def getNetProperties(self):
        print(self.network)
        print('No. of AE inputs :', self.nNetInput)
        print('No. of AE nodes :', self.nNodes)
        print('No. of AE parameters :', self.nParameters)

    def getNetParameters(self):
        print('Input weight: \n', self.network.linear.weight)
        print('Input bias: \n', self.network.linear.bias)
        print('Bias decoder: \n', self.network.biasDecoder)

    def netUpdateProperties(self):
        self.nNetInput = self.network.linear.in_features
        self.nNodes = self.network.linear.out_features
        self.nParameters = (self.network.linear.in_features * self.network.linear.out_features +
                            len(self.network.linear.bias.data) + len(self.network.biasDecoder))

    # ============================= evolving =============================
    def nodeGrowing(self, nNewNode=1, device=torch.device('cpu')):
        nNewNodeCurr = self.nNodes + nNewNode

        # grow node
        newWeight = nn.init.xavier_uniform_(torch.empty(nNewNode, self.nNetInput)).to(device)
        self.network.linear.weight.data = torch.cat((self.network.linear.weight.data,
                                                     newWeight), 0)  # grow input weights
        self.network.linear.bias.data = torch.cat((self.network.linear.bias.data,
                                                   torch.zeros(nNewNode).to(device)), 0)  # grow input bias
        self.network.linear.out_features = nNewNodeCurr
        del self.network.linear.weight.grad
        del self.network.linear.bias.grad

        self.netUpdateProperties()

    def nodePruning(self, pruneIdx, nPrunedNode=1):
        nNewNodeCurr = self.nNodes - nPrunedNode  # prune a node

        # prune node for current layer, output
        self.network.linear.weight.data = deleteRowTensor(self.network.linear.weight.data,
                                                          pruneIdx)  # prune input weights
        self.network.linear.bias.data = deleteRowTensor(self.network.linear.bias.data,
                                                        pruneIdx)  # prune input bias
        self.network.linear.out_features = nNewNodeCurr
        del self.network.linear.weight.grad
        del self.network.linear.bias.grad

        self.netUpdateProperties()

    def inputGrowing(self, nNewInput=1, device=torch.device('cpu')):
        nNewInputCurr = self.nNetInput + nNewInput

        # grow input weight
        newWeightNext = nn.init.xavier_uniform_(torch.empty(self.nNodes, nNewInput)).to(device)
        self.network.linear.weight.data = torch.cat((self.network.linear.weight.data, newWeightNext), 1)
        self.network.biasDecoder.data = torch.cat((self.network.biasDecoder.data, torch.zeros(nNewInput).to(device)), 0)

        del self.network.linear.weight.grad
        del self.network.biasDecoder.grad

        self.network.linear.in_features = nNewInputCurr
        self.netUpdateProperties()

    def inputPruning(self, pruneIdx, nPrunedNode=1):
        nNewInputCurr = self.nNetInput - nPrunedNode

        # prune input weight of next layer
        self.network.linear.weight.data = deleteColTensor(self.network.linear.weight.data, pruneIdx)
        self.network.biasDecoder.data = deleteRowTensor(self.network.biasDecoder.data, pruneIdx)

        del self.network.linear.weight.grad
        del self.network.biasDecoder.grad

        # update input features
        self.network.linear.in_features = nNewInputCurr
        self.netUpdateProperties()


class linearlizationlayer(nn.Module):
    def __init__(self):
        super(linearlizationlayer, self).__init__()

    def forward(self, x):
        return x


# ============================= Encoder Decoder =============================
class basicAE(nn.Module):
    def __init__(self, no_input, no_hidden):
        super(basicAE, self).__init__()
        # hidden layer
        self.linear = nn.Linear(no_input, no_hidden, bias=True)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.zero_()

        # decoder
        self.biasDecoder = nn.Parameter(torch.zeros(no_input))

    def forward(self, x, mode=1):
        # mode 1. encode, 2. decode
        if mode == 1:
            # encoder
            x = self.linear(x)
            x = self.activation(x)  # encoded output
            # x = self.sigmoid(x)

        if mode == 2:
            # decoder
            x = F.linear(x, self.linear.weight.t()) + self.biasDecoder
            x = self.activation(x)  # reconstructed input for end-to-end cnn
            # x = self.sigmoid(x)

        return x

class domain_classifier(nn.Module):
    def __init__(self,no_input, no_hidden):
        super(domain_classifier, self).__init__()
        self.linear = nn.Linear(no_input, no_hidden, bias=True)
        self.activation = nn.ReLU(inplace=True)
        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.zero_()
        self.classifier = nn.Linear(no_hidden, 2)

    def forward(self, x, alpha=1.0):
        reverse_feature = ReverseLayerF.apply(x, alpha)
        x = self.linear(reverse_feature)
        x = self.activation(x)
        x = self.classifier(x)
        x = F.softmax(x,1)
        return x

class mlpAePMNIST(nn.Module):
    def __init__(self, nNodes=384, nOutput=196):
        super(mlpAePMNIST, self).__init__()
        # hidden layer 1
        self.linear1 = nn.Linear(784, nNodes, bias=True)
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
        self.biasDecoder1 = nn.Parameter(torch.zeros(784))
        self.biasDecoder2 = nn.Parameter(torch.zeros(nNodes))

    def forward(self, x, mode=1):
        # mode 1. encode, 2. decode
        if mode == 1:
            # encoder
            x = x.view(x.size(0), -1)
            x = self.linear1(x)
            x = self.activation1(x)
            x = self.linear2(x)
            x = self.activation1(x)  # encoded output

        if mode == 2:
            # decoder
            x = F.linear(x, self.linear2.weight.t()) + self.biasDecoder2
            x = self.activation1(x)
            x = F.linear(x, self.linear1.weight.t()) + self.biasDecoder1
            x = self.activation2(x)  # reconstructed input for end-to-end cnn
            x = x.reshape(x.size(0), 1, 28, 28)

        return x


class simpleMPL(nn.Module):
    def __init__(self, nInput, nNodes=10, nOutput=5):
        super(simpleMPL, self).__init__()
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

    def forward(self, x, mode=1):
        # mode 1. encode, 2. decode
        if mode == 1:
            # encoder
            x = self.linear1(x)
            x = self.activation1(x)
            x = self.linear2(x)
            x = self.activation1(x)  # encoded output

        if mode == 2:
            # decoder
            x = F.linear(x, self.linear2.weight.t()) + self.biasDecoder2
            x = self.activation1(x)
            x = F.linear(x, self.linear1.weight.t()) + self.biasDecoder1
            x = self.activation2(x)  # reconstructed input for end-to-end

        return x


class singleMPL(nn.Module):
    def __init__(self, nInput, nNodes=10):
        super(singleMPL, self).__init__()
        # hidden layer 1
        self.linear1 = nn.Linear(nInput, nNodes, bias=True)
        nn.init.xavier_uniform_(self.linear1.weight)
        self.linear1.bias.data.zero_()

        # activation function
        self.activation1 = nn.ReLU(inplace=True)
        self.activation2 = nn.Sigmoid()

        # decoder
        self.biasDecoder1 = nn.Parameter(torch.zeros(nInput))

    def forward(self, x, mode=1):
        # mode 1. encode, 2. decode
        if mode == 1:
            # encoder
            x = self.linear1(x)
            x = self.activation1(x)  # encoded output

        if mode == 2:
            # decoder
            x = F.linear(x, self.linear1.weight.t()) + self.biasDecoder1
            x = self.activation2(x)  # reconstructed input for end-to-end

        return x


class ConvAeMNIST(nn.Module):
    def __init__(self):
        super(ConvAeMNIST, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 3 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)

        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)

        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)

        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2

        # self.t_conv1    = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)

        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)

        self.activation = nn.Sigmoid()

    def forward(self, x, mode=1):
        # mode 1. encode, 2. decode
        if mode == 1:
            ## encode ##
            # add hidden layers with relu activation function
            # and maxpooling after
            x = F.relu(self.conv1(x))
            x = self.pool(x)

            # add second hidden layer
            x = F.relu(self.conv2(x))
            x = self.pool(x)  # compressed representation
            x = x.view(x.size(0), -1)

        if mode == 2:
            ## decode ##
            # add transpose conv layers, with relu activation function
            # x = x.reshape(x.size(0), 4, 7, 7)
            x = x.reshape(x.size(0), 4, 7, 7)
            x = F.relu(self.t_conv1(x))

            # output layer (with sigmoid for scaling from 0 to 1)
            x = self.activation(self.t_conv2(x))

        return x

class ConvAeAM(nn.Module):
    def __init__(self):
        super(ConvAeAM, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 3 --> 16), 3x3 kernels
        self.conv1 = nn.Conv1d(768, 384, 1)
        self.conv2 = nn.Conv1d(384, 256, 1)

        ## decoder layers ##

        # self.t_conv1    = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv1 = nn.ConvTranspose1d(256, 384, 1)

        self.t_conv2 = nn.ConvTranspose1d(384, 768, 1)

        self.activation = nn.Sigmoid()

    def forward(self, x, mode=1):
        # mode 1. encode, 2. decode
        if mode == 1:
            ## encode ##
            # add hidden layers with relu activation function
            # and maxpooling after
            x = x.unsqueeze(2)
            x = F.relu(self.conv1(x))

            # add second hidden layer
            x = F.relu(self.conv2(x))
            x = x.view(x.size(0), -1)

        if mode == 2:
            ## decode ##
            # add transpose conv layers, with relu activation function
            # x = x.reshape(x.size(0), 4, 7, 7)
            x = x.reshape(x.size(0), 256, -1)
            x = F.relu(self.t_conv1(x))

            # output layer (with sigmoid for scaling from 0 to 1)
            x = self.activation(self.t_conv2(x))
            x = x.squeeze()

        return x
        
class ConvAeCIFAR(nn.Module):
    def __init__(self):
        super(ConvAeCIFAR, self).__init__()
        ## encoder layers ##
        self.conv1 = nn.Conv2d(3, 12, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(12, 24, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(24, 48, 4, stride=2, padding=1)

        ## decoder layers ##
        self.t_conv1 = nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1)
        self.t_conv2 = nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1)
        self.t_conv3 = nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1)
        self.activation = nn.Sigmoid()

    def forward(self, x, mode=1):
        # mode 1. encode, 2. decode
        if mode == 1:
            ## encode ##
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = x.view(x.size(0), -1)

        if mode == 2:
            ## decode ##
            # add transpose conv layers, with relu activation function
            x = x.reshape(x.size(0), 48, 4, 4)
            x = F.relu(self.t_conv1(x))
            x = F.relu(self.t_conv2(x))

            # output layer (with sigmoid for scaling from 0 to 1)
            x = self.activation(self.t_conv3(x))

        return x


# ============================= For LWF =============================
class ADCNoldtask():
    def __init__(self, taskId):
        # random seed control
        # np.random.seed(0)
        # torch.manual_seed(0)
        # random.seed(0)

        # initial network
        self.ADCNcnn = []
        self.ADCNae = []
        self.taskId = taskId

        # properties
        self.nInput = 0
        self.nOutput = 0
        self.nHiddenLayer = 1
        self.nHiddenNode = 0

    # ============================= forward pass =============================
    def forward(self, x, device=torch.device('cpu')):
        # encode decode in end-to-end manner
        # prepare model
        self.ADCNcnn = self.ADCNcnn.to(device)

        # prepare data
        x = x.to(device)

        # forward encoder CNN
        x = self.ADCNcnn(x)

        # feedforward from input layer to latent space, encode
        for iLayer, _ in enumerate(self.ADCNae):
            currnet = self.ADCNae[iLayer].network
            obj = currnet.train()
            obj = obj.to(device)
            x = obj(x)

        # feedforward from latent space to output layer, decode
        for iLayer in range(len(self.ADCNae) - 1, 0 - 1, -1):
            currnet = self.ADCNae[iLayer].network
            obj = currnet.train()
            obj = obj.to(device)
            x = obj(x, 2)

        # forward decoder CNN
        x = self.ADCNcnn(x, 2)

        return x


# ============================= For Comparison =============================
class simpleNet(nn.Module):
    def __init__(self):
        super(simpleNet, self).__init__()
        # hidden layer 1
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)

        # hidden layer 2
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)

        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)

        # output layer
        self.linear = nn.Linear(196, 10, bias=True)
        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.zero_()

        # activation function
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        # forward
        x = x.view(x.size(0), -1)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool(x)  # compressed representation
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x


class simpleNetPMNIST(nn.Module):
    def __init__(self):
        super(simpleNetPMNIST, self).__init__()
        # hidden layer
        self.linear1 = nn.Linear(784, 384, bias=True)
        nn.init.xavier_uniform_(self.linear1.weight)
        self.linear1.bias.data.zero_()

        self.linear2 = nn.Linear(384, 196, bias=True)
        nn.init.xavier_uniform_(self.linear2.weight)
        self.linear2.bias.data.zero_()

        self.linear3 = nn.Linear(196, 10, bias=True)
        nn.init.xavier_uniform_(self.linear3.weight)
        self.linear3.bias.data.zero_()

        # activation function
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        # forward
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)

        return x

class ConvDAEMNIST(nn.Module):
    def __init__(self):
        super(ConvDAEMNIST, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 3 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)

        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        # self.t_conv1    = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride=2)
        self.activation = nn.Sigmoid()

        ## fully connected layer ##
        # encoder
        self.linear = nn.Linear(196, 96,  bias=True)
        self.relu = nn.ReLU(inplace=True)
        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.zero_()
        # decoder
        self.biasDecoder = nn.Parameter(torch.zeros(196))

    def forward(self, x, mode=1):
        # mode 1. encode, 2. decode
        if mode == 1:
            ## encode ##
            # add hidden layers with relu activation function
            # and maxpooling after
            x = F.relu(self.conv1(x))
            x = self.pool(x)

            # add second hidden layer
            x = F.relu(self.conv2(x))
            x = self.pool(x)  # compressed representation
            x = x.view(x.size(0), -1)

            # add fully connected layer
            x = self.linear(x)
            x = self.relu(x)

        if mode == 2:
            ## decode ##
            # add fully connected layer
            x = F.linear(x, self.linear.weight.t()) + self.biasDecoder
            x = self.relu(x)                          # reconstructed input for end-to-end cnn

            # add transpose conv layers, with relu activation function
            # x = x.reshape(x.size(0), 4, 7, 7)
            x = x.reshape(x.size(0), 4, 7, 7)
            x = F.relu(self.t_conv1(x))

            # output layer (with sigmoid for scaling from 0 to 1)
            x = self.activation(self.t_conv2(x))

        return x


class DANNModel(nn.Module):
    def __init__(self):
        super(DANNModel, self).__init__()
        
        self.restored = False
        
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(1, 16, kernel_size=3, padding=1))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(16, 4, kernel_size=3, padding=1))
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(4*7*7, 96))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_fc2', nn.Linear(96, 10))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(4*7*7, 96))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(96, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):
        feature = self.feature(input_data)
        feature = feature.view(input_data.size(0), -1)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output

class DANN_MPL(nn.Module):
    def __init__(self, nInput, nNodes=10, nFeats=5, nOutput=5):
        super(DANN_MPL, self).__init__()
        
        self.restored = False
        
        self.feature = nn.Sequential()
        self.feature.add_module('fc_1', nn.Linear(nInput, nInput, bias=True))
        self.feature.add_module('fc_relu1', nn.ReLU(True))
        self.feature.add_module('fc_2', nn.Linear(nInput, nNodes, bias=True))
        self.feature.add_module('fc_relu2', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(nNodes, nFeats))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_fc2', nn.Linear(nFeats, nOutput))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(nNodes, nFeats))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(nFeats, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):
        feature = self.feature(input_data)
        feature = feature.view(input_data.size(0), -1)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output

import backbone
class DANN_office(nn.Module):
    def __init__(self, model_name, nHidden=1000, nOutput=31):
        super(DANN_office, self).__init__()
        
        self.feature = backbone.network_dict[model_name]()
        nFeat = self.feature.output_num()

        self.bottleneck = nn.Sequential(
            nn.Linear(nFeat, nHidden),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(nHidden, nOutput),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(nHidden, int(nHidden/2)),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(int(nHidden/2), int(nHidden/2)),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(int(nHidden/2), 2),
        )

    def forward(self, input_data, alpha):
        feature = self.feature(input_data)
        bottleneck = self.bottleneck(feature)

        reverse_feature = ReverseLayerF.apply(bottleneck, alpha)

        class_output = self.classifier(bottleneck)
        domain_output = self.discriminator(reverse_feature)

        return class_output, domain_output