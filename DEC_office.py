import numpy as np
import torch
import time
import os
import pickle

from dec_office31 import DEC
from torch.utils.data import DataLoader
from dataset import officeloader, officeloader_balanced

## load original image
def load_multiple_datastream(src, tgt, label_ratio, batch, balanced=True):
    if balanced:
        dataStreamSource = officeloader_balanced(src,label_ratio=label_ratio)
        dataStreamTarget = officeloader_balanced(tgt,label_ratio=0.0)
    else:
        dataStreamSource = officeloader(src,batch,label_ratio=label_ratio, shuffle=False, drop_last=True)
        dataStreamTarget = officeloader(tgt,batch,label_ratio=0.0, shuffle=False, drop_last=True)
    return dataStreamSource, dataStreamTarget


def DECMain(dataStreamSource, dataStreamTarget, lr, nBatch, nEpoch, noOfEpochInitializationCluster, listDriftSource, listDriftTarget, batch, device, typerun, params):

    Accuracy = []
    testingTime = []
    trainingTime = []
    F1_result = []
    precision_result = []
    recall_result = []
    nmi = []
    AccuracySource = []

    dec = DEC(31,500)
    
    # initialization
    print('Initialization Phase')
    start_initialization_train = time.time()
    b_model = dec.pretrain(dataStreamSource.labeledData, dataStreamSource.labeledLabel, 
                            noOfEpochInitializationCluster, batch, device, typerun, params.backbone)
    end_initialization_train = time.time()
    initialization_time      = end_initialization_train - start_initialization_train
    
    model = torch.load(b_model).to(dec.device)
    # testing initialization source labeled samples
    
    batchsize1 = int(len(dataStreamSource.dataset2)/nBatch)
    batchsize2 = int(dataStreamTarget.total_count/nBatch)

    dataset1 = DataLoader(dataStreamSource.dataset2, batch_size=batchsize1, shuffle=True)
    dataset2 = DataLoader(dataStreamTarget.unlabelset, batch_size=batchsize2, shuffle=True)

    # print(ADCNnet.ADCNcluster[0].centroids)
    for iBatch, (a,b) in enumerate(zip(dataset1, dataset2)):
        print(iBatch, '-th batch')

        # load data
        # Multistream
        batchIdx = iBatch + 1
        batchDataS = a[0]
        batchLabelS = [int(i) for i in a[1]]
        batchDataT = b[0]
        batchLabelT = [int(i) for i in b[1]]
        # if min(len(batchLabelS),len(batchLabelT)) < params.batch:
        #     print('Batch size wrt S & T:',len(batchLabelS), len(batchDataT))
        #     break

        # Multistream data if the data is listed in the list
        if iBatch in listDriftSource:
            dz = torch.rand(list(batchDataS.size())[0], list(batchDataS.size())[1], list(batchDataS.size())[2],
                            list(batchDataS.size())[3])
            batchDataS = torch.mul(dz, batchDataS) / torch.linalg.norm(batchDataS) * 224 * 224

        if iBatch in listDriftTarget:
            dz = torch.rand(list(batchDataT.size())[0], list(batchDataT.size())[1], list(batchDataT.size())[2],
                            list(batchDataT.size())[3])
            batchDataT = torch.mul(dz, batchDataT) / torch.linalg.norm(batchDataT) * 224 * 224

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

def run_dec_office(params):
    #Setting========
    TypeAblation=params.source+'2'+params.target
    # create folder to save performance file
    file_path = 'outputs/office/dec'
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
    
    paramstamp = ('DEC-office_' + TypeAblation+str(nRoundTest)+'Times_'+str(portion)+'Portion_'+str(noOfEpochInitializationCluster)
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

        print('----------------------------------------------')
        print(paramstamp)

        # Drift Location=============
        listDriftSource = [6]
        listDriftTarget = [7]

        nBatch = 10
        
        summaryacc = [] # added
        sumsourceacc = []
        sumf1 = []
        trtime = []
        nmiT = []

        for iRoundTest in range(nRoundTest):
            print("Running Test index :", iRoundTest, "Scheme :" + TypeAblation)
            dataStreamSource, dataStreamTarget = load_multiple_datastream(params.source, params.target,
                                                                            params.labeled_rate, params.batch,
                                                                            balanced=True)

            acc1, acc2, f1t, tr, nmi = DECMain(dataStreamSource, dataStreamTarget, lr, nBatch, nEpoch, 
                                            noOfEpochInitializationCluster, listDriftSource, listDriftTarget, 
                                            batch_size, device, TypeAblation, params)
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
            