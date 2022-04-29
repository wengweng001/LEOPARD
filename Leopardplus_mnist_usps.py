from utilsADCN import MnistLoaders,generalExtractorMnistUsps,evenly,listCollectS,UspsLoaders,amazonDataLoader,amazonDataLoaderOld,sourceAndTargetAmazonLoader,amazonDataLoaderRev,amazonDataLoaderUniform
from ACDCDataManipulator import DataManipulator
from model import simpleMPL, singleMPL
from LeopardBasic import ADCN
import numpy as np
import torch
import random
import time
from sklearn.metrics import precision_score, normalized_mutual_info_score, adjusted_rand_score, recall_score, f1_score
from torchvision import datasets, transforms
import pickle
import os
from data_load import usps, mnist


def load_multiple_datastream(TypeAblation, Mnist1, Mnist2, Usps1, Usps2, testingBatchSize, nEachClassSamples):
    if TypeAblation == 'MnistUsps':
        dataStreamSource=MnistLoaders(Mnist2, Mnist1, testingBatchSize=testingBatchSize, nEachClassSamples=nEachClassSamples)
        dataStreamTarget=generalExtractorMnistUsps(Usps1,Usps2)
    elif TypeAblation == 'UspsMnist':
        dataStreamSource = UspsLoaders(Usps2, Usps1, testingBatchSize=testingBatchSize, nEachClassSamples=nEachClassSamples)
        dataStreamTarget = generalExtractorMnistUsps(Mnist1, Mnist2)
    return dataStreamSource, dataStreamTarget

def run_leopard_mu(params):
    #Setting========
    TypeAblation=params.dataset
    alpha1 = params.alpha_kl
    alpha2 = params.alpha_dc


    # create folder to save performance file
    if TypeAblation == 'MnistUsps':
        file_path = 'outputs/Mnist2Usps/leopard+'
    elif TypeAblation == 'UspsMnist':
        file_path = 'outputs/Usps2Mnist/leopard+'
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    noOfEpochPrequential = params.epoch_kl
    nEpochKLL1 = params.epoch_dc
    noOfEpochInitializationCluster = params.epoch_clus_init  # number of epoch during cluster initialization #DEFAULT 1000
    nRoundTest = params.num_runs
    portion = params.labeled_rate
    device = params.device
    print(device)
    lr = params.learning_rate
    batch = params.batch
    nInitCluster=2 #setting number of initial cluster # DEFAULT 2

    paramstamp = ('Leopard++' + TypeAblation +str(nRoundTest)+'Times_'+str(portion)+'Portion_' + str(noOfEpochInitializationCluster) + 'initclusepoch_'
    +str(noOfEpochPrequential)+'ClusEpochs_'+str(nEpochKLL1)+'DCEpochs_'+str(alpha1)+
    'alphakl_'+str(alpha2)+'alphadc_' + str(batch) + 'Batchsize_'
    + str(lr) + 'lrSGD')
    resultfile = paramstamp+'.pkl'
    result_dir = file_path + '/' + resultfile
    if os.path.isfile(result_dir):
        try:
            w1, w2, w3, w4, w5, w6 = pickle.load(open(result_dir, 'rb'), encoding='utf-8')
    
            print('----------------------------------------------')
            print(paramstamp)
            print('epoch_clus\t', noOfEpochPrequential)
            print('epoch_dc\t', nEpochKLL1)
            print('alpha_clus\t', alpha1)
            print('alpha_dc\t', alpha2)
            print('Acc on T: %.4f +/- %.4f' % (np.mean(w1), np.std(w1)))
            print('F1 on T: %.4f +/- %.4f' % (np.mean(w5), np.std(w5)))
            print('Acc on S: %.4f +/- %.4f' % (np.mean(w2), np.std(w2)))
            print('Clusters: %.4f +/- %.4f' % (np.mean(w4), np.std(w4)))
    
            raise Exception('file exits')
        except:
            pass
    else:
        pass

    #data MNIST
    Mnist1 = mnist.MNIST('./data/mnist/', train=True, download=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                # transforms.Normalize((0.1307,), (0.3081,))
                            ]))
    Mnist1.dataset_size=Mnist1.data.shape[0]

    Mnist2 = mnist.MNIST('./data/mnist/', train=False, download=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                # transforms.Normalize((0.1307,), (0.3081,))
                            ]))
    Mnist2.dataset_size=Mnist2.data.shape[0]

    #data USPS
    Usps1 = usps.USPS_idx('./data/usps/', train=True, download=False,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    # transforms.Normalize((0.1307,), (0.3081,))
                                ]))
    Usps2 = usps.USPS('./data/usps/', train=False, download=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                # transforms.Normalize((0.1307,), (0.3081,))
                            ]))

    #Number of size data Source in total
    if TypeAblation == 'MnistUsps':
        nSizeSource = Mnist1.dataset_size + Mnist2.dataset_size
    elif TypeAblation == 'UspsMnist':
        nSizeSource = Usps1.dataset_size + Usps2.dataset_size

    # Drift Location=============
    listDriftSource = [35]
    listDriftTarget = [36]

    #Number of initial data Source
    nInitial=round(nSizeSource)*portion
    #Number of class
    nClass=10
    #Estimated number of samples for each class
    nEachClassSamples=round(nInitial/nClass)
    nBatch=65 #default 65 for Mnist and USPS
    testingBatchSize=round(nSizeSource*(1-portion)/nBatch)


    print('----------------------------------------------')
    print(paramstamp)

    summaryacc = [] # added
    nFeatures = []
    nClusters = []
    sumsourceacc = []
    sumf1 = []
    trtime = []

    for iRoundTest in range(nRoundTest):
        #Setting the source and target data based on portion
        print("Running Ablation index :", iRoundTest + 1, "Scheme :" + TypeAblation)
        
        dataStreamSource, dataStreamTarget = load_multiple_datastream(TypeAblation, Mnist1, Mnist2, Usps1, Usps2, testingBatchSize, nEachClassSamples)

        allMetrics = []  # Initial metrics
        allHistory = []

        ADCNnet = ADCN(dataStreamSource.nOutput, nInitCluster, alpha_kl=alpha1, alpha_dc=alpha2, LR=lr)  # Initialization of ADCN structure
    
        print(ADCNnet.ADCNcnn)
        print(ADCNnet.discriminator)

        Accuracy = []
        AccuracySource = []
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

        # initialization phase
        start_initialization_train = time.time()

        ADCNnet.initialization(dataStreamSource,
                            layerCount,
                            batchSize=batch,
                            epoch=noOfEpochInitializationCluster,
                            device=device, init=True)

        allegianceData = dataStreamSource.labeledData.clone()
        allegianceLabel = dataStreamSource.labeledLabel.clone()

        end_initialization_train = time.time()
        initialization_time = end_initialization_train - start_initialization_train

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

            start_train = time.time()

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
                                        batchSize=batch, epoch=3, device=device)

            # training data preparation
            previousBatchData = batchData.clone()
            # batchData, batchLabel = ADCNnet.trainingDataPreparation(batchData, batchLabel)

            # training
            if ADCNnet.driftStatus == 0 or ADCNnet.driftStatus == 2:  # only train if it is stable or drift

                ADCNnet.train_main(batchDataS, batchDataT, nEpochPreq=noOfEpochPrequential, nEpochKLL1=nEpochKLL1)

                ADCNnet.updateNetProperties()

                # update allegiance
                ADCNnet.updateAllegiance(allegianceData, allegianceLabel)

            end_train = time.time()
            training_time = end_train - start_train

            ## TESTING PART ##
            ADCNnet.testing(batchDataS,batchLabelS)
            AccuracySource.append(ADCNnet.accuracy)
            ADCNnet.testing(batchDataT, batchLabelT)
            AccuracyHistory.append(ADCNnet.accuracy)

            # if iBatch > 0:
            Y_pred = Y_pred + ADCNnet.predictedLabel.tolist()
            Y_true = Y_true + ADCNnet.trueClassLabel.tolist()

            # if iBatch > 0:
            # calculate performance
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
                            ADCNnet.nHiddenLayer, ADCNnet.nHiddenNode, ADCNnet.nCluster,
                            np.mean(AccuracySource)]

            allPerformanceHistory = [Iter, AccuracyHistory, nHiddenLayerHistory, nHiddenNodeHistory, nClusterHistory]
            
            # if (iBatch+1)%5==0:
            #     print('T:', AccuracyHistory)
            #     print('S',  AccuracySource)

        allMetrics.append(allPerformance)
        allHistory.append(allPerformanceHistory)

        filename = 'Results/' + TypeAblation + '/' + resultfile + 'Result{}'.format(iRoundTest)
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
        print('========== Performance'+TypeAblation+' ==========')
        print('Preq Accuracy: ', meanResults[0].item(), '(+/-)', stdResults[0].item())
        print('Source Accuracy: ', meanResults[11].item(), '(+/-)', stdResults[11].item())
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

        nFeatures.append(meanResults[9].item()) # added
        nClusters.append(meanResults[10].item())
        sumsourceacc.append(meanResults[11].item())
        summaryacc.append(meanResults[0].item())
        sumf1.append(meanResults[3].item())
        trtime.append(meanResults[6].item())

    print('========== ' + str(nRoundTest) + 'times results' + TypeAblation+' ==========')
    print('%.4f +/- %.4f' % (np.mean(summaryacc),np.std(summaryacc)))
    print('%.4f +/- %.4f' % (np.mean(sumsourceacc),np.std(sumsourceacc)))
    print('%.4f +/- %.4f' % (np.mean(sumf1),np.std(sumf1)))
    print('%.2f +/- %.2f' % (np.mean(nClusters),np.std(nClusters)))
    print('%.2f +/- %.2f' % (np.mean(trtime),np.std(trtime)))

    try:
        with open(result_dir, 'wb') as f:
            pickle.dump([summaryacc, sumsourceacc, nFeatures, nClusters, sumf1, trtime], f)
    except Exception as e:
        print('Save failed:{}'.format(e))
        
