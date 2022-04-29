import numpy as np
import torch
import time
from model import DANN_MPL
import torch.nn as nn

from sklearn.metrics import precision_score,recall_score,f1_score
from utilsADCN import evenly, listCollectS, sourceAndTargetAmazonLoader, amazonDataLoaderUniform
import os
import pickle

import pandas as pd

def pre_training(network, x, y, lr = 0.01, epoch = 1, batchSize = 16, device = torch.device('cpu')):
    
    nData = x.shape[0]
    x     = x.to(device)
    y     = y.to(device)

    # get optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(network.parameters(), lr = lr, momentum = 0.95, weight_decay = 0.00005)

    # prepare network
    network = network.train()
    network = network.to(device)
    
    for iEpoch in range(0, epoch):

        if iEpoch%10 == 0:
            print('Epoch: ', iEpoch)

        shuffled_indices = torch.randperm(nData)

        batchs = int(nData/batchSize)
        for i in range(batchs):
            # load data
            indices = shuffled_indices[i*batchSize:((i+1) * batchSize)]

            minibatch_xTrain = x[indices]
            minibatch_yTrain = y[indices]

            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # encode 
            outputs, _  = network(minibatch_xTrain, alpha=1.0)
            
            ## calculate loss
            # reconstruction loss
            loss = criterion(outputs, minibatch_yTrain)

            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # perform a single optimization step (parameter update)
            optimizer.step()
    
    return network


def training(network, batchDataS, batchDataT, lr = 0.01, epoch = 1, batchSize = 16, device = torch.device('cpu')):
    
    sourcelabel = torch.zeros(batchDataS.shape[0])
    targetlabel = torch.ones(batchDataT.shape[0])
    domain_label = torch.cat((sourcelabel, targetlabel),0)
    domain_label = domain_label.long()
    # domain_label = torch.zeros(domain_label.shape[0], 2).scatter_(1, domain_label.unsqueeze(1), 1)
    batchData = torch.cat((batchDataS, batchDataT), dim=0)

    nData           = batchData.shape[0]
    batchData       = batchData.to(device)
    domain_label    = domain_label.to(device)

    # get optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(network.parameters(), lr = lr, momentum = 0.95, weight_decay = 0.00005)

    # prepare network
    network = network.train()
    network = network.to(device)
    
    for iEpoch in range(0, epoch):

        # print('Epoch: ', iEpoch)

        shuffled_indices = torch.randperm(nData)
        batchs = int(nData/batchSize)
        for i in range(batchs):
            # load data
            indices = shuffled_indices[i*batchSize:((i+1) * batchSize)]

            minibatch_xTrain = batchData[indices]
            minibatch_yTrain = domain_label[indices]
            
            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # encode 
            cls_output, domain_output  = network(input_data=minibatch_xTrain, alpha=1.0)
            
            # ## calculate loss
            # loss = criterion(cls_output, class_true)
            
            # domain loss
            loss = criterion(domain_output,minibatch_yTrain)

            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # perform a single optimization step (parameter update)
            optimizer.step()
    
        if iEpoch%5 == 0:
            print('Loss: ', loss)

    return network


def DANN_Main(dataStreamSource, dataStreamTarget, lr, nBatch, nEpoch, noOfEpochInitializationCluster, 
            listDriftSource, listDriftTarget, batch, device):

    Accuracy = []
    AccuracySource = []
    testingTime = []
    trainingTime = []
    F1_result = []
    precision_result = []
    recall_result = []
    nmi = []
    AccuracySource = []

    Y_pred1 = []
    Y_true1 = []
    Y_pred2 = []
    Y_true2 = []

    nExtractedFeature = round(dataStreamSource.nInput/3)
    nFeatureClustering = round(dataStreamSource.nInput/10)

    net = DANN_MPL(nInput=dataStreamSource.nInput, nNodes=nExtractedFeature, 
                                nFeats=nFeatureClustering, nOutput=dataStreamSource.nOutput)
    
    # initialization
    print('Initialization Phase')
    start_initialization_train = time.time()
    net = pre_training(net, dataStreamSource.labeledData, dataStreamSource.labeledLabel, batchSize=batch, lr=lr, epoch = noOfEpochInitializationCluster, 
                               device = device)
    end_initialization_train = time.time()
    initialization_time      = end_initialization_train - start_initialization_train
    
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
            dz = torch.rand(list(batchDataS.size())[0], list(batchDataS.size())[1])
            # batchDataS = torch.mul(dz, batchDataS) / torch.linalg.norm(batchDataS)
            batchDataS = torch.mul(dz, batchDataS) / torch.norm(batchDataS)
        if iBatch in listDriftTarget:
            dz = torch.rand(list(batchDataT.size())[0], list(batchDataT.size())[1])
            # batchDataT = torch.mul(dz, batchDataT) / torch.linalg.norm(batchDataT)
            batchDataT = torch.mul(dz, batchDataT) / torch.norm(batchDataT)
        
        # training
        start_train = time.time()
        net = training(net, batchDataS, batchDataT, epoch=nEpoch, batchSize=batch, lr=lr, device = device)
        end_train     = time.time()
        training_time = end_train - start_train

        # testing target samples
        start_test     = time.time()
        net.eval()
        alpha = 0
        preds, _ = net(batchDataT.to(device),alpha)
        pred_cls = preds.data.max(1)[1]
        end_test     = time.time()
        testing_time = end_test - start_test
        correct        = (pred_cls.cpu() == batchLabelT).sum().item()
        accuracy       = 100*correct/(pred_cls.cpu() == batchLabelT).shape[0]  # 1: correct, 0: wrong
        Y_pred2 = Y_pred2 + pred_cls.tolist()
        Y_true2 = Y_true2 + batchLabelT.tolist()
        
        # calculate performance
        Accuracy.append(accuracy)
        testingTime.append(testing_time)
        trainingTime.append(training_time)
        F1_result.append(f1_score(Y_true2, Y_pred2, average='weighted'))
        precision_result.append(precision_score(Y_true2, Y_pred2, average='weighted'))
        recall_result.append(recall_score(Y_true2, Y_pred2, average='weighted'))

        # testing source samples
        start_test     = time.time()
        net.eval()
        alpha = 0
        preds, _ = net(batchDataS.to(device),alpha)
        pred_cls = preds.data.max(1)[1]
        end_test     = time.time()
        testing_time = end_test - start_test
        correct        = (pred_cls.cpu() == batchLabelS).sum().item()
        accuracy       = 100*correct/(pred_cls.cpu() == batchLabelS).shape[0]  # 1: correct, 0: wrong
        Y_pred1 = Y_pred1 + pred_cls.tolist()
        Y_true1 = Y_true1 + batchLabelS.tolist()

        # calculate performance
        AccuracySource.append(accuracy)
        
    print('\n')
    print('=== Performance result ===')
    print('Accuracy Target: ',np.mean(Accuracy),'(+/-)',np.std(Accuracy))
    print('Accuracy Source: ',np.mean(AccuracySource),'(+/-)',np.std(AccuracySource))
    print('F1 score: ',np.mean(F1_result),'(+/-)',np.std(F1_result))
    print('Precision: ',np.mean(precision_result),'(+/-)',np.std(precision_result))
    print('Recall: ',np.mean(recall_result),'(+/-)',np.std(recall_result))
    print('Testing Time: ',np.mean(testingTime),'(+/-)',np.std(testingTime))
    print('Training Time: ',np.mean(trainingTime) + initialization_time,'(+/-)',np.std(trainingTime))

    return np.mean(Accuracy), np.mean(AccuracySource), np.mean(F1_result), np.mean(trainingTime)

def run_dann(params):
    # create folder to save performance file
    file_path = 'outputs/AM/dann'
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    df = []
    #========Setting========
    Source = ['AM1', 'AM1', 'AM1', 'AM1', 'AM2', 'AM2', 'AM2', 'AM2', 'AM3', 'AM3', 'AM3', 'AM3', 'AM4', 'AM4', 'AM4', 'AM4', 'AM5', 'AM5', 'AM5', 'AM5']
    Target = ['AM2', 'AM3', 'AM4', 'AM5', 'AM1', 'AM3', 'AM4', 'AM5', 'AM1', 'AM2', 'AM4', 'AM5', 'AM1', 'AM2', 'AM3', 'AM5', 'AM1', 'AM2', 'AM3', 'AM4']
            
    for s, t in zip(Source,Target):

        Init = 'Run'
        TypeRun=s+t+Init
        # print(TypeRun)

        nRoundTest = params.num_runs
        nEpoch = params.epochs                                # number of epoch during prequential (KL and L1)
        noOfEpochInitializationCluster = params.epoch_clus_init     # number of epoch during cluster initialization #DEFAULT 1000
        portion = params.labeled_rate
        device = params.device
        # print(device)
        lr = params.learning_rate
        batch_size = params.batch
            
        # Drift Location=============
        listDriftSource = [5]
        listDriftTarget = [6]

        paramstamp = ('DANN-' + TypeRun+str(nRoundTest)+'Times_'+str(portion)+'Portion_'+str(noOfEpochInitializationCluster)
                        +'initEpochs' + str(nEpoch)+'Epochs' + str(batch_size) + 'Batchsize_'
                        + str(lr) + 'lrSGD')
        resultfile = paramstamp+'.pkl'
        result_dir = file_path + '/' + resultfile
        if os.path.isfile(result_dir):
            try:
                w1, w2, w3, w4 = pickle.load(open(result_dir, 'rb'), encoding='utf-8')
        
                print('----------------------------------------------')
                print(paramstamp)
                print('Acc on T: %.4f +/- %.4f' % (np.mean(w1), np.std(w1)))
                print('Acc on S: %.4f +/- %.4f' % (np.mean(w2), np.std(w2)))
                print('F1 on T: %.4f +/- %.4f' % (np.mean(w3), np.std(w3)))
                df.append([TypeRun,round(np.mean(w1),4),round(np.std(w1),4),round(np.mean(w2),4),round(np.std(w2),4)])
        
                continue
            except:
                pass
        else:
            print('----------------------------------------------')
            print(paramstamp)

            # # load the training and test datasets
            data = sourceAndTargetAmazonLoader(s,t)
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

            summaryacc = [] # added
            sumsourceacc = []
            sumf1 = []
            trtime = []
            nmiT = []

            for iRoundTest in range(nRoundTest):
                print("Running Test index :", iRoundTest, "Scheme :" + TypeRun)
                acc1, acc2, f1t, tr = DANN_Main(dataStreamSource, dataStreamTarget, lr, nBatch, nEpoch, 
                                                noOfEpochInitializationCluster, listDriftSource, 
                                                listDriftTarget, batch_size, device)
                summaryacc.append(acc1)
                sumsourceacc.append(acc2)
                sumf1.append(f1t)
                trtime.append(tr)
            
            print('========== ' + str(nRoundTest) + 'times results ' + TypeRun+' ==========')
            # print(summaryacc)
            # print(sumf1)
            print('%.4f +/- %.4f' % (np.mean(summaryacc),np.std(summaryacc)))
            print('%.4f +/- %.4f' % (np.mean(sumsourceacc),np.std(sumsourceacc)))
            print('%.4f +/- %.4f' % (np.mean(sumf1),np.std(sumf1)))
            print('%.2f +/- %.2f' % (np.mean(trtime),np.std(trtime)))

            try:
                with open(result_dir, 'wb') as f:
                    pickle.dump([summaryacc, sumsourceacc, sumf1, trtime], f)
            except Exception as e:
                print('Save failed:{}'.format(e))
                
    df = pd.DataFrame(df, columns=['experiment','target acc','target std','source acc','source std'])
    df.to_csv('./csv/dann.csv', index=False)