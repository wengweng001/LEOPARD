from utilsADCN import MnistLoaders,generalExtractorMnistUsps,evenly,listCollectS,UspsLoaders,amazonDataLoader,amazonDataLoaderOld,sourceAndTargetAmazonLoader
from ACDCDataManipulator import DataManipulator
from model import simpleMPL, singleMPL
from ADCNBasic import ADCN, ADCNMPL
import numpy as np
import torch
import random
import time
from sklearn.metrics import precision_score, normalized_mutual_info_score, adjusted_rand_score, recall_score, f1_score
from torchvision import datasets, transforms
import pickle
import os

#========Setting========
Source = 'AM1'
Target = 'AM2'
Init = 'Init'
TypeRun= Source+Target+Init
default=True

if default: # This setting parameters are used in all experiments
    nRoundTest = 5  # default 5 or 1 if you one to run 1 time
    portion = 0.1;  # portion of labeled data in the warm up phase #DEFAULT 0.1, 0.2 UP LEAD TO ERROR DUE TO LACK OF LABELED DATA
    nInitCluster = 2  # setting number of initial cluster # DEFAULT 2
    noOfEpochInitializationCluster = 100  # number of epoch during cluster initialization #DEFAULT 1000
    noOfEpochPrequential = 5  # number of epoch during prequential (CNN and Auto Encoder) #DEFAULT 5
    nEpochKLL1 = 1  # number of epoch during prequential (KL and L1) # DEFAULT 1
    # Drift Location=============
    listDriftSource = [5]
    listDriftTarget = [6]
else: # This setting one to test
    nRoundTest = 10  # default 5 or 1 if you one to run 1 time
    portion = 0.1;  # portion of labeled data in the warm up phase #DEFAULT 0.1, 0.2 UP LEAD TO ERROR DUE TO LACK OF LABELED DATA
    nInitCluster = 2  # setting number of initial cluster # DEFAULT 2
    noOfEpochInitializationCluster = 100  # number of epoch during cluster initialization #DEFAULT 1000
    noOfEpochPrequential = 5  # number of epoch during prequential (CNN and Auto Encoder) #DEFAULT 5
    nEpochKLL1 = 1  # number of epoch during prequential (KL and L1) # DEFAULT 1
    # Drift Location=============
    listDriftSource = [5]
    listDriftTarget = [6]

 # Beauty vs Magazine
# np.random.seed(42)

for iRoundTest in range(nRoundTest):

    torch.manual_seed(iRoundTest)

    # Beauty1
    data = sourceAndTargetAmazonLoader(Source,Target)

    # #Magazine
    # magazineAmazonData = DataManipulator()
    # magazineAmazonData.load_amazon_review_magazine_subscription()

    #Number of Source Sample
    # nSizeSource = magazineAmazonData.number_samples()
    nSizeSource = data.S.number_samples()
    #Number of initial data Source
    nInitial=round(nSizeSource)*portion

    #Number of class
    nClass=5

    #Estimated number of samples for each class
    nEachClassSamples=int(nInitial/nClass)
    nBatch=20 #default 65 for Mnist and USPS
    testingBatchSize=round(nSizeSource*(1-portion)/nBatch)

    print("Running Test index :", iRoundTest, "Scheme :" + TypeRun)
    dataStreamSource = amazonDataLoaderOld(data=data.S, nEachClassSamples=nEachClassSamples)
    dataStreamTarget = amazonDataLoaderOld(data=data.T, nEachClassSamples=None)
    device = torch.device('cpu')

    nHidNodeExtractor = dataStreamSource.nInput
    nExtractedFeature = round(dataStreamSource.nInput/3)
    nFeatureClustering = round(dataStreamSource.nInput/10)

    allMetrics = []  # Initial metrics
    allHistory = []

    ADCNnet = ADCNMPL(dataStreamSource.nOutput, nInput = nExtractedFeature, nHiddenNode=nFeatureClustering)  # Initialization of ADCN structure
    ADCNnet.ADCNcnn = simpleMPL(dataStreamSource.nInput, nNodes=nHidNodeExtractor, nOutput=nExtractedFeature)
    # ADCNnet.ADCNcnn = singleMPL(dataStreamSource.nInput, 100)
    ADCNnet.desiredLabels = [0,1,2,3,4]

    Accuracy = []
    testingTime = []
    trainingTime = []

    prevBatchData = []

    Y_pred = []
    Y_true = []
    Iter = []

    # for figure
    AccuracyHistory = []
    nHiddenLayerHistory = []
    nHiddenNodeHistory = []
    nClusterHistory = []

    # network evolution
    nHiddenNode = []
    nHiddenLayer = []
    nCluster = []
    layerCount = 0


    nLabeled = 1;
    layerGrowing = True;
    nodeEvolution = True;
    clusterGrowing = True;
    lwfLoss = True;
    clusteringLoss = True;
    trainingBatchSize = 16;

    # initialization phase
    start_initialization_train = time.time()

    ADCNnet.initialization(dataStreamSource,
                           layerCount,
                           batchSize=trainingBatchSize,
                           epoch=noOfEpochInitializationCluster,
                           device=device)


    allegianceData = dataStreamSource.labeledData.clone()
    allegianceLabel = dataStreamSource.labeledLabel.clone()

    end_initialization_train = time.time()
    initialization_time = end_initialization_train - start_initialization_train

    nSampleDistributionSource = evenly(dataStreamSource.unlabeledData.shape[0], nBatch)
    nSampleDistributionTarget = evenly(dataStreamTarget.unlabeledData.shape[0], nBatch)
    listOfSource = listCollectS(nSampleDistributionSource)
    listOfTarget = listCollectS(nSampleDistributionTarget)

    for iBatch in range(0, nBatch):
    # for iBatch in [0,1,63,64]:

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
            dz = torch.rand(list(batchDataS.size())[0], list(batchDataS.size())[1])
            batchDataS = torch.mul(dz, batchDataS) / torch.linalg.norm(batchDataS)

        if iBatch in listDriftTarget:
            dz = torch.rand(list(batchDataT.size())[0], list(batchDataT.size())[1])
            batchDataT = torch.mul(dz, batchDataT) / torch.linalg.norm(batchDataT)

        batchData = torch.cat((batchDataS, batchDataT), dim=0)
        # shuffledList = np.arange(batchData.shape[0])
        # np.random.shuffle(shuffledList)
        shuffledList = torch.randperm(batchData.shape[0])
        batchData = batchData[shuffledList, :]  # Target and Source data are shuffled together

        start_train = time.time()#

        layerGrowing = True
        if iBatch > 0 and layerGrowing:
            # drift detection
            ADCNnet.driftDetection(batchData, previousBatchData)

            if ADCNnet.driftStatus == 2:
                # grow layer if drift is confirmed driftStatus == 2
                ADCNnet.layerGrowing()
                layerCount += 1

                # initialization phase
                ADCNnet.initialization(dataStreamSource, layerCount,
                                       batchSize=trainingBatchSize, epoch=3, device=device)

        # training data preparation
        previousBatchData = batchData.clone()
        # batchData, batchLabel = ADCNnet.trainingDataPreparation(batchData, batchLabel)

        # training
        if ADCNnet.driftStatus == 0 or ADCNnet.driftStatus == 2:  # only train if it is stable or drift

            ADCNnet.fitLeo(batchDataS, batchDataT, nEpochPreq=noOfEpochPrequential, nEpochKLL1=nEpochKLL1)

            ADCNnet.updateNetProperties()

            # update allegiance
            ADCNnet.updateAllegiance(allegianceData, allegianceLabel)

        end_train = time.time()
        training_time = end_train - start_train

        # TESTING PART

        ADCNnet.testing(batchDataT, batchLabelT)
        # if iBatch > 0:
        Y_pred = Y_pred + ADCNnet.predictedLabel.tolist()
        Y_true = Y_true + ADCNnet.trueClassLabel.tolist()

        # if iBatch > 0:
        # calculate performance
        AccuracyHistory.append(ADCNnet.accuracy)
        # AccuracyHistory.append(ADCNnet.accuracy)
        testingTime.append(ADCNnet.testingTime)
        trainingTime.append(training_time)
        # testingLoss.append(ADCNnet.testingLoss)

        # calculate network evolution
        nHiddenLayer.append(ADCNnet.nHiddenLayer)
        nHiddenNode.append(ADCNnet.nHiddenNode)
        nCluster.append(ADCNnet.nCluster)

        nHiddenLayerHistory.append(ADCNnet.nHiddenLayer)
        nHiddenNodeHistory.append(ADCNnet.nHiddenNode)
        nClusterHistory.append(ADCNnet.nCluster)

        Iter.append(iBatch  + 1)

        if iBatch % 1 == 0 or iBatch == 0:
            print('Accuracy: ', np.mean(AccuracyHistory))

        allPerformance = [np.mean(AccuracyHistory), adjusted_rand_score(Y_true, Y_pred),
                          normalized_mutual_info_score(Y_true, Y_pred),
                          f1_score(Y_true, Y_pred, average='weighted'),
                          precision_score(Y_true, Y_pred, average='weighted'),
                          recall_score(Y_true, Y_pred, average='weighted'),
                          (np.mean(trainingTime) + initialization_time), np.mean(testingTime),
                          ADCNnet.nHiddenLayer, ADCNnet.nHiddenNode, ADCNnet.nCluster]

        allPerformanceHistory = [Iter, AccuracyHistory, nHiddenLayerHistory, nHiddenNodeHistory, nClusterHistory]

    allMetrics.append(allPerformance)
    allHistory.append(allPerformanceHistory)

    filename = 'Results/' + Source + '/' + Target + '/' + TypeRun + 'Result{}'.format(iRoundTest)
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            raise
    with open(filename, 'wb') as f:
        pickle.dump([allMetrics, allHistory], f)

    print("==============Results, Accummulated Average===============")
    meanResults = np.round_(np.mean(allMetrics, 0), decimals=2)
    stdResults = np.round_(np.std(allMetrics, 0), decimals=2)

    print('\n')
    print('========== Performance'+TypeRun+' ==========')
    print('Preq Accuracy: ', meanResults[0].item(), '(+/-)', stdResults[0].item())
    print('ARI: ', meanResults[1].item(), '(+/-)', stdResults[1].item())
    print('NMI: ', meanResults[2].item(), '(+/-)', stdResults[2].item())
    print('F1 score: ', meanResults[3].item(), '(+/-)', stdResults[3].item())
    print('Precision: ', meanResults[4].item(), '(+/-)', stdResults[4].item())
    print('Recall: ', meanResults[5].item(), '(+/-)', stdResults[5].item())
    print('Training time: ', meanResults[6].item(), '(+/-)', stdResults[6].item())
    print('Testing time: ', meanResults[7].item(), '(+/-)', stdResults[7].item())
    print('\n')

    print('========== Network ==========')
    print('Number of hidden layers: ', meanResults[8].item(), '(+/-)', stdResults[8].item())
    print('Number of features: ', meanResults[9].item(), '(+/-)', stdResults[9].item())
    print('Number of clusters: ', meanResults[10].item(), '(+/-)', stdResults[10].item())