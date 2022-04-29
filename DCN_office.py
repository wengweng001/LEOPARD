import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import time

from DAE_conv import ConvDAE
from utilsADCN import clusteringLoss
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

def pre_training(network, km, x, lr = 0.01, epoch = 1, batchSize = 16, device = torch.device('cpu')):
    
    nData = x.shape[0]
    x     = x.to(device)

    # get optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(network.parameters(), lr = lr, momentum = 0.95, weight_decay = 0.00005)

    # prepare network
    network = network.train()
    network = network.to(device)
    
    for iEpoch in range(0, epoch):

        if iEpoch%20 == 0:
            print('Epoch: ', iEpoch)

        shuffled_indices = torch.randperm(nData)

        batchs = int(nData/batchSize)
        for i in range(batchs):
            # load data
            indices = shuffled_indices[i*batchSize:((i+1) * batchSize)]

            minibatch_xTrain = x[indices]

            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # encode 
            latent  = network(minibatch_xTrain)
            
            # decode
            outputs = network(latent, 2)
            
            ## calculate loss
            # reconstruction loss
            loss = criterion(outputs, minibatch_xTrain)

            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # perform a single optimization step (parameter update)
            optimizer.step()
    
    with torch.no_grad():
        feature = network(x)
    
    return network, feature.detach().clone().to('cpu')

def training(network, km, batchDataS, batchDataT, lr = 0.01, epoch = 1, batchSize = 16, device = torch.device('cpu')):
    
    batchData = torch.cat((batchDataS, batchDataT), dim=0)
    shuffledList = torch.randperm(batchData.shape[0])
    batchData = batchData[shuffledList, :]  # Target and Source data are shuffled together

    nData = batchData.shape[0]
    x     = batchData.to(device)
    
    # get optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(network.parameters(), lr = lr, momentum = 0.95, weight_decay = 0.00005)

    # prepare network
    network = network.train()
    network = network.to(device)
    
    for iEpoch in range(0, epoch):

        print('Epoch: ', iEpoch)

        shuffled_indices = torch.randperm(nData)
        batchs = int(nData/batchSize)
        for i in range(batchs):
            # load data
            indices = shuffled_indices[i*batchSize:((i+1) * batchSize)]

            minibatch_xTrain = x[indices]
            
            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # encode 
            latent  = network(minibatch_xTrain)
            
            # decode
            outputs = network(latent, 2)
            
            ## calculate loss
            # reconstruction loss
            loss = criterion(outputs, minibatch_xTrain)
            
            # clustering loss
            pred = km.predict(latent.detach().clone().to('cpu').numpy())
            oneHotClusters = F.one_hot(torch.tensor(pred).long(), 
                                        num_classes = km.n_clusters).float().to(device)
            centroids      = torch.tensor(km.cluster_centers_).float().to(device)
            loss.add_(0.01/2,clusteringLoss(latent, oneHotClusters, centroids))

            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # perform a single optimization step (parameter update)
            optimizer.step()
    
    with torch.no_grad():
        feature1 = network(batchDataS.to(device))
        feature2 = network(batchDataT.to(device))
    
    return network, feature1.detach().clone().to('cpu'), feature2.detach().clone().to('cpu')

# MIT License

# Copyright (c) 2019 Sajjad Salaria

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

def infer_cluster_labels(km, actual_labels):
    inferred_labels = {}
    
    for i in range(km.n_clusters):
#         print('i', i)
        # find index of points in cluster
        labels = []
        index  = np.where(km.labels_ == i)
        
#         print('index: ', index)
        # append actual labels for each point in cluster
        labels.append(actual_labels[index])
#         print('index: ', index)

        # determine most common label
        if len(labels[0]) == 1:
            counts = np.bincount(labels[0])
        else:
            counts = np.bincount(np.squeeze(labels))
#         print('counts', counts)
        # assign the cluster to a value in the inferred_labels dictionary
        if len(counts) == 0:
            print('counts is empty')
            continue
#             counts = prev_counts
#             pdb.set_trace()
            
        if np.argmax(counts) in inferred_labels:
            # append the new number to the existing array at this slot
            inferred_labels[np.argmax(counts)].append(i)
            prev_counts = counts
        else:
            # create a new array in this slot
            inferred_labels[np.argmax(counts)] = [i]
            prev_counts = counts

        #print(labels)
        #print('Cluster: {}, label: {}'.format(i, np.argmax(counts)))
        
    return inferred_labels

def infer_data_labels(X_labels, cluster_labels):
  # empty array of len(X)
    predicted_labels = np.zeros(len(X_labels)).astype(np.uint8)

    for i, cluster in enumerate(X_labels):
        for key, value in cluster_labels.items():
            if cluster in value:
                predicted_labels[i] = key
                
    return predicted_labels

# test the infer_cluster_labels() and infer_data_labels() functions
# train
# cluster_labels = infer_cluster_labels(kmeans, labeledLabel)

# test
# X_clusters       = kmeans.predict(X)
# predicted_labels = infer_data_labels(X_clusters, cluster_labels)
# print predicted_labels[:20]
# print Y[:20]


def testing(km, x, predictedLabel, label):
    # testing
    start_test          = time.time()
    km.predict(x.numpy())
    end_test            = time.time()
    testingTime    = end_test - start_test
    correct             = (predictedLabel == label.numpy()).sum().item()
    accuracy       = 100*correct/(predictedLabel == label.numpy()).shape[0]  # 1: correct, 0: wrong
    trueClassLabel = label.numpy()

    return accuracy, testingTime, trueClassLabel

def kmeanMain(dataStreamSource, dataStreamTarget, lr, nBatch, nEpoch, noOfEpochInitializationCluster, listDriftSource, listDriftTarget, batch, device, params):

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

    batchsize1 = int(len(dataStreamSource.dataset2)/nBatch)
    batchsize2 = int(dataStreamTarget.total_count/nBatch)
    n_clusters = min(batchsize1,batchsize2) - min(batchsize1,batchsize2) % 5
    print('Kmeans clusters:', n_clusters)

    net = ConvDAE(base_net=params.backbone)
    km  = KMeans(
    n_clusters=n_clusters, init='random',
    n_init=31, max_iter=50, 
    tol=1e-04, random_state=0
)
    
    # initialization
    print('Initialization Phase')
    start_initialization_train = time.time()
    
    net, latent = pre_training(net, km, dataStreamSource.labeledData, batchSize=batch, lr=lr, epoch = noOfEpochInitializationCluster, 
                               device = device)
    km.fit(latent.numpy())
    
    # assign label to a cluster
    cluster_labels0 = infer_cluster_labels(km, dataStreamSource.labeledLabel.numpy())
    end_initialization_train = time.time()
    initialization_time      = end_initialization_train - start_initialization_train
    
    # training accuracy
    X_clusters     = km.predict(latent.numpy())
    predictedLabel = infer_data_labels(X_clusters, cluster_labels0)
    correct  = (predictedLabel == dataStreamSource.labeledLabel.numpy()).sum().item()
    accuracy = 100*correct/latent.shape[0]  # 1: correct, 0: wrong
    print('Pre-Training accuracy on source: ', accuracy)
    
    km.max_iter = 1
    
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
        if min(len(batchLabelS),len(batchLabelT)) < params.batch:
            print('Batch size wrt S & T:',len(batchLabelS), len(batchDataT))
            break

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
        net, latent1, latent2 = training(net, km, batchDataS, batchDataT, epoch=nEpoch, batchSize=batch, lr=lr, device = device)
        
        end_train     = time.time()
        training_time = end_train - start_train

        # testing target samples
        start_test     = time.time()
        X_clusters     = km.fit_predict(latent2.numpy())
        predictedLabel = infer_data_labels(X_clusters, cluster_labels0)
        end_test     = time.time()
        testing_time = end_test - start_test

        # start_test     = time.time()
        # predictedLabel = km.fit_predict(latent2.numpy())
        # end_test     = time.time()
        # testing_time = end_test - start_test
        
        # calculate accuracy
        correct        = (predictedLabel == batchLabelT).sum().item()
        accuracy       = 100*correct/(predictedLabel == batchLabelT).shape[0]  # 1: correct, 0: wrong

        Y_pred2 = Y_pred2 + predictedLabel.tolist()
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
        X_clusters     = km.fit_predict(latent1.numpy())
        predictedLabel = infer_data_labels(X_clusters, cluster_labels0)
        end_test     = time.time()
        testing_time = end_test - start_test

        # start_test     = time.time()
        # predictedLabel = km.fit_predict(latent1.numpy())
        # end_test     = time.time()
        # testing_time = end_test - start_test

        # calculate accuracy
        correct        = (predictedLabel == batchLabelS).sum().item()
        accuracy       = 100*correct/(predictedLabel == batchLabelS).shape[0]  # 1: correct, 0: wrong

        Y_pred1 = Y_pred1 + predictedLabel.tolist()
        Y_true1 = Y_true1 + batchLabelS

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

def run_dcn_office(params):
    #Setting========
    TypeAblation=params.source+'2'+params.target
    # create folder to save performance file
    file_path = 'outputs/office/dcn'
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
    
    paramstamp = ('DCN-office_' + TypeAblation +str(nRoundTest)+'Times'+str(portion)+'Portion'+str(noOfEpochInitializationCluster)
                    +'initEpochs' + str(nEpoch)+'Epochs')
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
    
            raise Exception('Result exists')
        except:
            pass
    else:
        # Drift Location=============
        listDriftSource = [6]
        listDriftTarget = [7]

        nBatch = 10

        #Number of class
        nClass=31


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
            acc1, acc2, f1t, tr = kmeanMain(dataStreamSource, dataStreamTarget, lr, nBatch, nEpoch, noOfEpochInitializationCluster, listDriftSource, listDriftTarget, batch, device, params)
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