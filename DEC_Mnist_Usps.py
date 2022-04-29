import numpy as np
import torch
import time
from torchvision import datasets, transforms

from utilsADCN import clusteringLoss,MnistLoaders,generalExtractorMnistUsps,evenly,listCollectS,UspsLoaders,amazonDataLoader,amazonDataLoaderOld,sourceAndTargetAmazonLoader,amazonDataLoaderRev,amazonDataLoaderUniform
import os
import pickle

from data_load import usps, mnist
from dec_mnist import DEC


def load_multiple_datastream(TypeAblation, Mnist1, Mnist2, Usps1, Usps2, testingBatchSize, nEachClassSamples):
    if TypeAblation == 'MnistUsps':
        dataStreamSource=MnistLoaders(Mnist2, Mnist1, testingBatchSize=testingBatchSize, nEachClassSamples=nEachClassSamples)
        dataStreamTarget=generalExtractorMnistUsps(Usps1,Usps2)
    elif TypeAblation == 'UspsMnist':
        dataStreamSource = UspsLoaders(Usps2, Usps1, testingBatchSize=testingBatchSize, nEachClassSamples=nEachClassSamples)
        dataStreamTarget = generalExtractorMnistUsps(Mnist1, Mnist2)
    return dataStreamSource, dataStreamTarget


def DECMain(dataStreamSource, dataStreamTarget, lr, nBatch, nEpoch, noOfEpochInitializationCluster, listDriftSource, listDriftTarget, batch, device, typerun):

    Accuracy = []
    testingTime = []
    trainingTime = []
    F1_result = []
    precision_result = []
    recall_result = []
    nmi = []
    AccuracySource = []

    dec = DEC(10, 96)
    
    # initialization
    print('Initialization Phase')
    start_initialization_train = time.time()
    b_model = dec.pretrain(dataStreamSource.labeledData, dataStreamSource.labeledLabel, 
                            noOfEpochInitializationCluster, batch, device, typerun)
    end_initialization_train = time.time()
    initialization_time      = end_initialization_train - start_initialization_train
    
    model = torch.load(b_model).to(dec.device)
    
    # testing initialization source labeled samples
    
    nSampleDistributionSource = evenly(dataStreamSource.unlabeledData.shape[0], nBatch)
    nSampleDistributionTarget = evenly(dataStreamTarget.unlabeledData.shape[0], nBatch)
    listOfSource = listCollectS(nSampleDistributionSource)
    listOfTarget = listCollectS(nSampleDistributionTarget)

    for iBatch in range(0, nBatch):
        print(iBatch,'-th batch')
        
        # load data
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

        # update
        start_train = time.time()

        batchData = torch.cat((batchDataS, batchDataT), dim=0)
        shuffledList = np.arange(batchData.shape[0])
        np.random.shuffle(shuffledList)
        batchData = batchData[shuffledList, :]  # Target and Source data are shuffled together
        model = dec.trainae(model, batchData, nEpoch, lr)

        # encoder decoder
        model, metrics_S, metrics_T = dec.train(batchDataS, batchLabelS, batchDataT, batchLabelT, nEpoch, lr, model)

        end_train     = time.time()
        training_time = end_train - start_train
        
        # calculate performance on target samples
        Accuracy.append(metrics_T[0])
        testingTime.append(metrics_T[4])
        trainingTime.append(training_time)
        F1_result.append(metrics_T[1])
        precision_result.append(metrics_T[2])
        recall_result.append(metrics_T[3])
        nmi.append(metrics_T[5])
        print(metrics_T[0], metrics_S[0])
        # calculate performance on source samples
        AccuracySource.append(metrics_S[0])
        
    print('\n')
    print('=== Performance result ===')
    print('Accuracy Target: ',np.mean(Accuracy),'(+/-)',np.std(Accuracy))
    print('Accuracy Source: ',np.mean(AccuracySource),'(+/-)',np.std(AccuracySource))
    print('F1 score: ',np.mean(F1_result),'(+/-)',np.std(F1_result))
    print('Precision: ',np.mean(precision_result),'(+/-)',np.std(precision_result))
    print('Recall: ',np.mean(recall_result),'(+/-)',np.std(recall_result))
    print('Testing Time: ',np.mean(testingTime),'(+/-)',np.std(testingTime))
    print('Training Time: ',np.mean(trainingTime) + initialization_time,'(+/-)',np.std(trainingTime))

    return np.mean(Accuracy), np.mean(AccuracySource), np.mean(F1_result), np.mean(trainingTime), np.mean(nmi)

def run_dec_mu(params):
    #Setting========
    TypeAblation=params.dataset
    # create folder to save performance file
    if TypeAblation == 'MnistUsps':
        file_path = 'outputs/Mnist2Usps/dec'
    elif TypeAblation == 'UspsMnist':
        file_path = 'outputs/Usps2Mnist/dec'
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    nRoundTest = params.num_runs
    nEpoch = params.epochs                                # number of epoch during prequential (KL and L1)
    noOfEpochInitializationCluster = params.epoch_clus_init     # number of epoch during cluster initialization #DEFAULT 1000
    portion = params.labeled_rate
    device = params.device
    print(device)
    lr = params.learning_rate
    batch_size = params.batch
    
    paramstamp = ('DEC-' + TypeAblation+str(nRoundTest)+'Times_'+str(portion)+'Portion_'+str(noOfEpochInitializationCluster)
                    +'initEpochs_' + str(nEpoch)+'Epochs_' + str(batch_size) + 'Batchsize_'
                    + str(lr) + 'lrSGD') 
    resultfile = paramstamp+'.pkl'
    result_dir = file_path + '/' + resultfile
    if os.path.isfile(result_dir):
        try:
            w1, w2, w3, w4, w5 = pickle.load(open(result_dir, 'rb'), encoding='utf-8')
    
            print('----------------------------------------------')
            print(paramstamp)
            print('Acc on T: %.4f +/- %.4f' % (np.mean(w1), np.std(w1)))
            print('Acc on S: %.4f +/- %.4f' % (np.mean(w2), np.std(w2)))
            print('F1 on T: %.4f +/- %.4f' % (np.mean(w3), np.std(w3)))
            print('NMI on T: %.4f +/- %.4f' % (np.mean(w5), np.std(w5)))
    
            raise Exception('Result exists')
        except:
            pass
    else:
        pass

    print('----------------------------------------------')
    print(paramstamp)

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

    summaryacc = [] # added
    sumsourceacc = []
    sumf1 = []
    trtime = []
    nmiT = []

    for iRoundTest in range(nRoundTest):
        print("Running Test index :", iRoundTest, "Scheme :" + TypeAblation)
        dataStreamSource, dataStreamTarget = load_multiple_datastream(TypeAblation, Mnist1, Mnist2, Usps1, Usps2, testingBatchSize, nEachClassSamples)
        acc1, acc2, f1t, tr, nmi = DECMain(dataStreamSource, dataStreamTarget, lr, nBatch, nEpoch, 
                                        noOfEpochInitializationCluster, listDriftSource, listDriftTarget, 
                                        batch_size, device, TypeAblation)
        summaryacc.append(acc1)
        sumsourceacc.append(acc2)
        sumf1.append(f1t)
        trtime.append(tr)
        nmiT.append(nmi)
    
    print('========== ' + str(nRoundTest) + 'times results ' + TypeAblation +' ==========')
    # print(summaryacc)
    # print(sumf1)
    print('%.4f +/- %.4f' % (np.mean(summaryacc),np.std(summaryacc)))
    print('%.4f +/- %.4f' % (np.mean(sumsourceacc),np.std(sumsourceacc)))
    print('%.4f +/- %.4f' % (np.mean(sumf1),np.std(sumf1)))
    print('%.2f +/- %.2f' % (np.mean(trtime),np.std(trtime)))
    print('%.4f +/- %.4f' % (np.mean(nmiT),np.std(nmiT)))

    try:
        with open(result_dir, 'wb') as f:
            pickle.dump([summaryacc, sumsourceacc, sumf1, trtime, nmiT], f)
    except Exception as e:
        print('Save failed:{}'.format(e))
        