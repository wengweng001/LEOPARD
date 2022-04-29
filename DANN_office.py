import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from torch.utils.data import TensorDataset
import time
import math
from dann import DANN_office
import os
import pickle

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

def testing(dataset, net, device):
    data = TensorDataset(dataset.unlabeledData, dataset.unlabeledLabel)
    dataloader = torch.utils.data.DataLoader(dataset=data,batch_size=128,shuffle=False)
    n_total = 0
    n_correct = 0
    net.eval()
    alpha = 0
    for i, (x,y) in enumerate(dataloader):
        with torch.no_grad():
            preds, _ = net(x.to(device),alpha)
            pred_cls = preds.data.max(1)[1]
            # calculate accuracy
            correct        = (pred_cls.cpu() == y.cpu()).sum().item()
            # accuracy       = 100*correct/(pred_cls.cpu() == y.cpu()).shape[0]  # 1: correct, 0: wrong

            n_correct   += correct
            n_total     += y.shape[0]
        if i == 10:
            break
    accuracy = n_correct/n_total
    net.train()
    return accuracy

def pre_training(network, x, y, TypeAblation, labeled_rate,iRoundTest, lr = 0.01, epoch = 1, batchSize = 16, device = torch.device('cpu'), testdata=None):
    
    nData = x.shape[0]
    x     = x.to(device)
    y     = y.long()
    y     = y.to(device)

    # get optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(network.parameters(), lr = lr, momentum = 0.95, weight_decay = 0.00005)

    # prepare network
    network = network.train()
    network = network.to(device)

    loss_trace = []
    acc_trace  = []

    for iEpoch in range(0, epoch):

        if iEpoch%10 == 0:
            print('Epoch: ', iEpoch)

        shuffled_indices = torch.randperm(nData)

        batchs = int(nData/batchSize)
        for i in range(batchs):
            # load data
            indices = shuffled_indices[i*batchSize:((i+1) * batchSize)]

            minibatch_xTrain = x[indices,:,:,:]
            minibatch_yTrain = y[indices]

            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # encode 
            outputs, _  = network(minibatch_xTrain, alpha=1.0)
            
            ## calculate loss
            # reconstruction loss
            loss = criterion(outputs, minibatch_yTrain)
            loss_trace.append(loss.detach().cpu().clone())

            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # perform a single optimization step (parameter update)
            optimizer.step()

            # accuracy = testing(testdata,network,device)
            # acc_trace.append(accuracy*100)

    # plt.subplot(2,1,1)
    # plt.plot(np.arange(len(loss_trace)),loss_trace, 'b-')
    # plt.subplot(2,1,2)
    # plt.plot(np.arange(len(acc_trace)),acc_trace, 'r-')
    # plt.savefig('initialization.jpg')
    # plt.show()
    # np.save('Init acc T -{}-{}-run{}.npy'.format(TypeAblation, labeled_rate,iRoundTest), acc_trace) 
    # np.save('Init loss -{}-{}-run{}.npy'.format(TypeAblation, labeled_rate,iRoundTest), loss_trace) 

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

    return network

def dannMain(dataStreamSource, dataStreamTarget, lr, nBatch, nEpoch, noOfEpochInitializationCluster, 
                listDriftSource, listDriftTarget, batch, device, TypeAblation, labeled_rate, iRoundTest, params):

    allMetrics = []  # Initial metrics
    allHistory = []
    
    Accuracy = []
    testingTime = []
    trainingTime = []
    F1_result = []
    precision_result = []
    recall_result = []

    prevBatchData = []

    Y_pred1 = []
    Y_true1 = []
    Y_pred2 = []
    Y_true2 = []
    Iter = []
    
    # for figure
    AccuracyHistory = []
    AccuracySource = []

    net = DANN_office(params.backbone)
    
    # initialization
    print('Initialization Phase')
    start_initialization_train = time.time()
    
    net = pre_training(net, dataStreamSource.labeledData, dataStreamSource.labeledLabel, TypeAblation, labeled_rate,iRoundTest, batchSize=batch, lr=lr, epoch = noOfEpochInitializationCluster, 
                               device = device, testdata=dataStreamTarget)

    end_initialization_train = time.time()
    initialization_time      = end_initialization_train - start_initialization_train
    
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
        #     pass

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
        
        # encoder decoder
        net = training(net, batchDataS, batchDataT, epoch=nEpoch, batchSize=batch, lr=lr, device = device)

        
        end_train     = time.time()
        training_time = end_train - start_train

        # testing target samples
        start_test     = time.time()
        with torch.no_grad():
            net.eval()
            for i in range(math.ceil(batchDataT.size(0)/batch)):
                alpha = 0
                batch_data = batchDataT[i*batch:(i+1)*batch].to(device)
                preds, _ = net(batch_data,alpha)
                pred_cl = preds.data.max(1)[1]
                if i == 0:
                    pred_cls=pred_cl
                else:
                    pred_cls=torch.cat((pred_cls,pred_cl),0)
        end_test     = time.time()    
        testing_time = end_test - start_test
        # calculate accuracy
        correct        = (pred_cls.cpu().numpy() == batchLabelT).sum().item()
        accuracy       = 100*correct/(pred_cls.cpu().numpy() == batchLabelT).shape[0]  # 1: correct, 0: wrong

        Y_pred2 = Y_pred2 + pred_cls.tolist()
        Y_true2 = Y_true2 + batchLabelT
        
        # calculate performance
        Accuracy.append(accuracy)
        testingTime.append(testing_time)
        trainingTime.append(training_time)
        F1_result.append(f1_score(Y_true2, Y_pred2, average='weighted'))
        precision_result.append(precision_score(Y_true2, Y_pred2, average='weighted'))
        recall_result.append(recall_score(Y_true2, Y_pred2, average='weighted'))

        # testing source samples
        start_test     = time.time()
        with torch.no_grad():
            net.eval()
            for i in range(math.ceil(batchDataS.size(0)/batch)):
                alpha = 0
                batch_data = batchDataS[i*batch:(i+1)*batch].to(device)
                preds, _ = net(batch_data,alpha)
                pred_cl = preds.data.max(1)[1]
                if i == 0:
                    pred_cls=pred_cl
                else:
                    pred_cls=torch.cat((pred_cls,pred_cl),0)
        end_test     = time.time()
        testing_time = end_test - start_test

        # calculate accuracy
        correct        = (pred_cls.cpu().numpy() == batchLabelS).sum().item()
        accuracy       = 100*correct/(pred_cls.cpu().numpy() == batchLabelS).shape[0]  # 1: correct, 0: wrong

        Y_pred1 = Y_pred1 + pred_cls.tolist()
        Y_true1 = Y_true1 + batchLabelS

        # calculate performance
        AccuracySource.append(accuracy)
    
    # plt.plot(np.arange(len(AccuracySource)),AccuracySource)
    # plt.savefig('source_acc.jpg')
    # plt.show()
    # plt.plot(np.arange(len(Accuracy)),Accuracy)
    # plt.savefig('target_acc.jpg')
    # plt.show()
    # np.save('Accuracy-T-{}-{}-run{}.npy'.format(TypeAblation, labeled_rate,iRoundTest), Accuracy) 
    # np.save('Accuracy-S-{}-{}-run{}.npy'.format(TypeAblation, labeled_rate,iRoundTest), AccuracySource) 
        
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

def run_dann_office(params):
    #Setting========
    TypeAblation=params.source+'2'+params.target
    # create folder to save performance file
    file_path = 'outputs/office/dann'
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    nRoundTest = params.num_runs
    nEpoch = params.epochs                                # number of epoch during prequential (KL and L1)
    noOfEpochInitializationCluster = params.epoch_clus_init     # number of epoch during cluster initialization #DEFAULT 1000
    portion = params.labeled_rate
    device = params.device
    print(device)
    lr = params.learning_rate
    batch = params.batch
    
    paramstamp = ('DANN-office_' + TypeAblation +str(nRoundTest)+'Times'+str(portion)+'Portion'+str(noOfEpochInitializationCluster)
                    +'initEpochs' + str(nEpoch)+'Epochs' + '-Seed' + str(params.seed))
    resultfile = paramstamp+'.pkl'
    result_dir = file_path + '/' + resultfile
    if os.path.isfile(result_dir):
    # if False:
        w1, w2, w3, w4 = pickle.load(open(result_dir, 'rb'), encoding='utf-8')

        print('----------------------------------------------')
        print(paramstamp)
        print('Acc on T: %.4f +/- %.4f' % (np.mean(w1), np.std(w1)))
        print('Acc on S: %.4f +/- %.4f' % (np.mean(w2), np.std(w2)))
        print('F1 on T: %.4f +/- %.4f' % (np.mean(w3), np.std(w3)))
    else:
        
        # Drift Location=============
        listDriftSource = [6]
        listDriftTarget = [7]

        nBatch = 10


        print('----------------------------------------------')
        print(paramstamp)

        summaryacc = [] # added
        sumsourceacc = []
        sumf1 = []
        trtime = []

        for iRoundTest in range(nRoundTest):
            print("Running Test index :", iRoundTest, "Scheme :" + TypeAblation)
            dataStreamSource, dataStreamTarget = load_multiple_datastream(params.source, params.target,
                                                                            params.labeled_rate, params.batch,
                                                                            balanced=True)
            acc1, acc2, f1t, tr = dannMain(dataStreamSource, dataStreamTarget, lr, nBatch, nEpoch, noOfEpochInitializationCluster, 
                                            listDriftSource, listDriftTarget, batch, device, TypeAblation, params.labeled_rate, iRoundTest, params)
            summaryacc.append(acc1)
            sumsourceacc.append(acc2)
            sumf1.append(f1t)
            trtime.append(tr)
        
        print('========== ' + str(nRoundTest) + 'times results ' + TypeAblation +' ==========')
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