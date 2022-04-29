from torchvision import datasets, transforms
import torch
import numpy as np
from data_load import usps, mnist
from utilsADCN import MnistLoaders,generalExtractorMnistUsps,evenly,listCollectS,UspsLoaders

from model import ConvDAEMNIST

def load_multiple_datastream(TypeAblation, Mnist1, Mnist2, Usps1, Usps2, testingBatchSize, nEachClassSamples):
    if TypeAblation == 'MnistUsps':
        dataStreamSource=MnistLoaders(Mnist2, Mnist1, testingBatchSize=testingBatchSize, nEachClassSamples=nEachClassSamples)
        dataStreamTarget=generalExtractorMnistUsps(Usps1,Usps2)
    elif TypeAblation == 'UspsMnist':
        dataStreamSource = UspsLoaders(Usps2, Usps1, testingBatchSize=testingBatchSize, nEachClassSamples=nEachClassSamples)
        dataStreamTarget = generalExtractorMnistUsps(Mnist1, Mnist2)
    return dataStreamSource, dataStreamTarget

net = ConvDAEMNIST()

#data MNIST
Mnist1 = mnist.MNIST('./data/mnist/', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                        ]))
Mnist1.dataset_size=Mnist1.data.shape[0]

Mnist2 = mnist.MNIST('./data/mnist/', train=False, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                        ]))
Mnist2.dataset_size=Mnist2.data.shape[0]

#data USPS
Usps1 = usps.USPS_idx('./data/usps/', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                            ]))
Usps2 = usps.USPS('./data/usps/', train=False, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                        ]))

TypeAblation = 'UspsMnist'

#Number of size data Source in total
if TypeAblation == 'MnistUsps':
    nSizeSource = Mnist1.dataset_size + Mnist2.dataset_size
elif TypeAblation == 'UspsMnist':
    nSizeSource = Usps1.dataset_size + Usps2.dataset_size

# Drift Location=============
listDriftSource = [35]
listDriftTarget = [36]

#Number of initial data Source
nInitial=round(nSizeSource)*0.1
#Number of class
nClass=10
#Estimated number of samples for each class
nEachClassSamples=round(nInitial/nClass)
nBatch=65 #default 65 for Mnist and USPS
testingBatchSize=round(nSizeSource*(1-0.1)/nBatch)


    
dataStreamSource, dataStreamTarget = load_multiple_datastream(TypeAblation, Mnist1, Mnist2, Usps1, Usps2, testingBatchSize, nEachClassSamples)

nSampleDistributionSource = evenly(dataStreamSource.unlabeledData.shape[0], nBatch)
nSampleDistributionTarget = evenly(dataStreamTarget.unlabeledData.shape[0], nBatch)
listOfSource = listCollectS(nSampleDistributionSource)
listOfTarget = listCollectS(nSampleDistributionTarget)

# print(ADCNnet.ADCNcluster[0].centroids)
for iBatch in range(0, nBatch):
    print(iBatch, '-th batch')
    
    # load data
    # Multistream
    batchIdx = iBatch + 1
    batchDataS = dataStreamSource.unlabeledData[listOfSource[iBatch]]
    batchLabelS = dataStreamSource.unlabeledLabel[listOfSource[iBatch]]
    batchDataT = dataStreamTarget.unlabeledData[listOfTarget[iBatch]]
    batchLabelT = dataStreamTarget.unlabeledLabel[listOfTarget[iBatch]]

    # Multistream data if the data is listed in the list
    if iBatch in listDriftSource:
        dz = torch.rand(list(batchDataS.size())[0], list(batchDataS.size())[1], list(batchDataS.size())[2],
                        list(batchDataS.size())[3])
        batchDataS = torch.mul(dz, batchDataS) / torch.linalg.norm(batchDataS) * 28 * 28

    if iBatch in listDriftTarget:
        dz = torch.rand(list(batchDataT.size())[0], list(batchDataT.size())[1], list(batchDataT.size())[2],
                        list(batchDataT.size())[3])
        batchDataT = torch.mul(dz, batchDataT) / torch.linalg.norm(batchDataT) * 28 * 28

    batchData = torch.cat((batchDataS, batchDataT), dim=0)
    shuffledList = np.arange(batchData.shape[0])
    np.random.shuffle(shuffledList)
    batchData = batchData[shuffledList, :]  # Target and Source data are shuffled together
