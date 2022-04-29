import numpy as np
import torch
import time

from utilsADCN import evenly, listCollectS, sourceAndTargetAmazonLoader, amazonDataLoaderUniform
import os
import pickle

from dec import DEC



def DECMain(dataStreamSource, dataStreamTarget, lr, nBatch, nEpoch, noOfEpochInitializationCluster, 
            listDriftSource, listDriftTarget, batch, device):

    Accuracy = []
    testingTime = []
    trainingTime = []
    F1_result = []
    precision_result = []
    recall_result = []
    nmi = []
    AccuracySource = []

    nHidNodeExtractor = dataStreamSource.nInput
    nExtractedFeature = round(dataStreamSource.nInput/3)

    dec = DEC(5,nHidNodeExtractor)
    
    # initialization
    print('Initialization Phase')
    start_initialization_train = time.time()
    b_model = dec.pretrain(dataStreamSource.labeledData, dataStreamSource.labeledLabel, 
                            nExtractedFeature, noOfEpochInitializationCluster, batch, device)
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
            dz = torch.rand(list(batchDataS.size())[0], list(batchDataS.size())[1])
            # batchDataS = torch.mul(dz, batchDataS) / torch.linalg.norm(batchDataS)
            batchDataS = torch.mul(dz, batchDataS) / torch.norm(batchDataS)
        if iBatch in listDriftTarget:
            dz = torch.rand(list(batchDataT.size())[0], list(batchDataT.size())[1])
            # batchDataT = torch.mul(dz, batchDataT) / torch.linalg.norm(batchDataT)
            batchDataT = torch.mul(dz, batchDataT) / torch.norm(batchDataT)
        
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

def run_dec(params):
    # create folder to save performance file
    file_path = 'outputs/AM/dec'
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    #========Setting========
    Source = ['AM1', 'AM1', 'AM1', 'AM1', 'AM2', 'AM2', 'AM2', 'AM2', 'AM3', 'AM3', 'AM3', 'AM3', 'AM4', 'AM4', 'AM4', 'AM4', 'AM5', 'AM5', 'AM5', 'AM5',]
    Target = ['AM5', 'AM3', 'AM4', 'AM2', 'AM1', 'AM3', 'AM4', 'AM5', 'AM1', 'AM2', 'AM4', 'AM5', 'AM1', 'AM2', 'AM3', 'AM5', 'AM1', 'AM2', 'AM3', 'AM4',]
            
    for s, t in zip(Source,Target):

        Init = 'Run'
        TypeRun=s+t+Init
        print(TypeRun)

        nRoundTest = params.num_runs
        nEpoch = params.epochs                                # number of epoch during prequential (KL and L1)
        noOfEpochInitializationCluster = params.epoch_clus_init     # number of epoch during cluster initialization #DEFAULT 1000
        portion = params.labeled_rate
        device = params.device
        print(device)
        lr = params.learning_rate
        batch_size = params.batch
            
        # Drift Location=============
        listDriftSource = [5]
        listDriftTarget = [6]

        paramstamp = ('DEC-' + TypeRun+str(nRoundTest)+'Times_'+str(portion)+'Portion_'+str(noOfEpochInitializationCluster)
                        +'initEpochs' + str(nEpoch)+'Epochs' + str(batch_size) + 'Batchsize_'
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
                acc1, acc2, f1t, tr, nmi = DECMain(dataStreamSource, dataStreamTarget, lr, nBatch, nEpoch, 
                                                noOfEpochInitializationCluster, listDriftSource, 
                                                listDriftTarget, batch_size, device)
                summaryacc.append(acc1)
                sumsourceacc.append(acc2)
                sumf1.append(f1t)
                trtime.append(tr)
                nmiT.append(nmi)
            
            print('========== ' + str(nRoundTest) + 'times results ' + TypeRun+' ==========')
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
                