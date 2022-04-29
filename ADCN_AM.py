import os
import pickle
import time

import numpy as np
import torch
from sklearn.metrics import precision_score, normalized_mutual_info_score, adjusted_rand_score, recall_score, f1_score

from ADCNBasic_minibatch import ADCNMPL
from model import simpleMPL
from utilsADCN import evenly, listCollectS, amazonDataLoaderOld, sourceAndTargetAmazonLoader, amazonDataLoaderUniform


def run_acdn(params):
    # create folder to save performance file
    file_path = 'outputs/AM/adcn'
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    #========Setting========
    Source = ['AM1', 'AM1', 'AM1', 'AM1', 'AM2', 'AM2', 'AM2', 'AM2', 'AM3', 'AM3', 'AM3', 'AM3', 'AM4', 'AM4', 'AM4', 'AM4', 'AM5', 'AM5', 'AM5', 'AM5',]
    Target = ['AM2', 'AM3', 'AM4', 'AM5', 'AM1', 'AM3', 'AM4', 'AM5', 'AM1', 'AM2', 'AM4', 'AM5', 'AM1', 'AM2', 'AM3', 'AM5', 'AM1', 'AM2', 'AM3', 'AM4',]

    for s, t in zip(Source,Target):
        # ========Setting========
        noOfEpochPrequential = params.epoch_kl                      # number of epoch during prequential (CNN and Auto Encoder)
        nEpochKLL1 = params.epoch_dc                                # number of epoch during prequential (KL and L1)
        noOfEpochInitializationCluster = params.epoch_clus_init     # number of epoch during cluster initialization #DEFAULT 1000
        nRoundTest = params.num_runs
        portion = params.labeled_rate
        device = params.device
        print(device)
        lr = params.learning_rate
        batch = params.batch
        # Drift Location
        listDriftSource = [5]
        listDriftTarget = [6]

        Init = 'Run'
        TypeRun= s+t+Init

        paramstamp = ('ADCN-' + TypeRun + str(nRoundTest) + 'Times_' + str(portion) + 'Portion_' + 
            str(noOfEpochInitializationCluster) + 'initEpoch_' + str(
            noOfEpochPrequential) + 'ClusEpochs_' + str(nEpochKLL1) + 'L1Epochs_' + str(batch) + 'Batchsize_'
            + str(lr) + 'lrSGD')
        resultfile = paramstamp + '.pkl'
        result_dir = file_path + '/' + resultfile
        print('----------------------------------------------')
        print(paramstamp)
        if os.path.isfile(result_dir):
            w1, w2, w3, w4, w5, w6 = pickle.load(open(result_dir, 'rb'), encoding='utf-8')
            print('epoch_clus\t', noOfEpochPrequential)
            print('epoch_dc\t', nEpochKLL1)
            print('Acc on T: %.4f +/- %.4f' % (np.mean(w1), np.std(w1)))
            print('F1 on T: %.4f +/- %.4f' % (np.mean(w5), np.std(w5)))
            print('Acc on S: %.4f +/- %.4f' % (np.mean(w2), np.std(w2)))
            print('Clusters: %.4f +/- %.4f' % (np.mean(41), np.std(w4)))
            continue
        else:
            pass

        data = sourceAndTargetAmazonLoader(s,t)
        torch.manual_seed(2)
        nSizeSource = data.S.number_samples()
        #Number of initial data Source
        nInitial=round(nSizeSource)*portion

        #Number of class
        nClass=5

        #Estimated number of samples for each class
        nEachClassSamples=int(nInitial/nClass)
        nBatch=20 #default 65 for Mnist and USPS
        testingBatchSize=round(nSizeSource*(1-portion)/nBatch)

        dataStreamSource = amazonDataLoaderUniform(data=data.S, nEachClassSamples=nEachClassSamples) # load data from each class by chosen percent
        dataStreamTarget = amazonDataLoaderUniform(data=data.T, nEachClassSamples=None)
        # torch.random.seed()

        label1 = dataStreamSource.labeledLabel.numpy()
        label2 = dataStreamSource.unlabeledLabel.numpy()
        a = np.concatenate((label1,label2),axis=0)
        unique_elements, counts_elements = np.unique(a, return_counts=True)
        print("Frequency of unique values of the said array:")
        print(np.asarray((unique_elements, counts_elements)))
        
        label3 = dataStreamTarget.labeledLabel.numpy()
        label4 = dataStreamTarget.unlabeledLabel.numpy()
        b = np.concatenate((label3,label4),axis=0)
        unique_elements, counts_elements = np.unique(b, return_counts=True)
        print("Frequency of unique values of the said array:")
        print(np.asarray((unique_elements, counts_elements)))


        summaryacc = [] # added
        nFeatures = []
        nClusters = []
        sumsourceacc = []
        sumf1 = []
        trtime = []

        for iRoundTest in range(nRoundTest):
            print("Running Test index :", iRoundTest, "Scheme :" + TypeRun)
            nHidNodeExtractor = dataStreamSource.nInput
            nExtractedFeature = round(dataStreamSource.nInput/3)
            nFeatureClustering = round(dataStreamSource.nInput/10)

            allMetrics = []  # Initial metrics
            allHistory = []
            AccuracySource = []


            ADCNnet = ADCNMPL(dataStreamSource.nOutput, nInput = nExtractedFeature, nHiddenNode=nFeatureClustering, LR=lr)  # Initialization of ADCN structure
            ADCNnet.ADCNcnn = simpleMPL(dataStreamSource.nInput, nNodes=nHidNodeExtractor, nOutput=nExtractedFeature)
            # ADCNnet.ADCNcnn = singleMPL(dataStreamSource.nInput, 100)
            ADCNnet.desiredLabels = [0,1,2,3,4]

            print(ADCNnet.ADCNcnn)

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

            # initialization phase
            start_initialization_train = time.time()

            ADCNnet.initialization(dataStreamSource,
                                layerCount,
                                batchSize=batch,
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
                    # batchDataS = torch.mul(dz, batchDataS) / torch.linalg.norm(batchDataS)
                    batchDataS = torch.mul(dz, batchDataS) / torch.norm(batchDataS)
                if iBatch in listDriftTarget:
                    dz = torch.rand(list(batchDataT.size())[0], list(batchDataT.size())[1])
                    # batchDataT = torch.mul(dz, batchDataT) / torch.linalg.norm(batchDataT)
                    batchDataT = torch.mul(dz, batchDataT) / torch.norm(batchDataT)

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
                                            batchSize=batch, epoch=3, device=device)

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
                ADCNnet.testing(batchDataS,batchLabelS)
                AccuracySource.append(ADCNnet.accuracy)
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
                                ADCNnet.nHiddenLayer, ADCNnet.nHiddenNode, ADCNnet.nCluster,
                                np.mean(AccuracySource)]

                allPerformanceHistory = [Iter, AccuracyHistory, nHiddenLayerHistory, nHiddenNodeHistory, nClusterHistory]

            allMetrics.append(allPerformance)
            allHistory.append(allPerformanceHistory)

            # filename = 'Results/' + s + '/' + t + '/' + TypeRun + 'Result{}'.format(iRoundTest)
            # if not os.path.exists(os.path.dirname(filename)):
            #     try:
            #         os.makedirs(os.path.dirname(filename))
            #     except OSError as exc:  # Guard against race condition
            #         raise
            # with open(filename, 'wb') as f:
            #     pickle.dump([allMetrics, allHistory], f)

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

            nFeatures.append(meanResults[9].item()) # added
            nClusters.append(meanResults[10].item())
            sumsourceacc.append(meanResults[11].item())
            summaryacc.append(meanResults[0].item())
            sumf1.append(meanResults[3].item())
            trtime.append(meanResults[7].item())

        print('\n')
        print(summaryacc)
        print(nFeatures)
        print(nClusters)
        print(sumf1)
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