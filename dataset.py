import torch
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, STL10
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import copy
import pandas as pd
import torch.utils.data as data_utils

class officeloader_balanced(object):
    def __init__(self, name_dataset, label_ratio = 0.0):
        # root dir (local pc or colab)
        root_dir = "data/office/%s/images" % name_dataset

        __datasets__ = ["amazon", "dslr", "webcam"]

        if name_dataset not in __datasets__:
            raise ValueError("must introduce one of the three datasets in office")

        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        self.nOutput = 31
        print('Number of output: ', self.nOutput)

        dataset = datasets.ImageFolder(root=root_dir,
                                    transform=data_transforms)

        self.total_count = len(dataset)

        if label_ratio == 0:
            self.get_all(dataset)
        else:
            self.get_label_unlabel(dataset, label_ratio)

    def get_label_unlabel(self, dataset, label_rate):
        n_per_class = round(self.total_count*label_rate/self.nOutput)
        unique_elements, counts_elements = np.unique(dataset.targets, return_counts=True)
        print('Class distribution:\n',counts_elements)

        targets = np.asarray(dataset.targets)
        for class_i in range(self.nOutput):
            idx = (targets==class_i).nonzero()[0]
            A, B = np.split(np.random.permutation(idx), [n_per_class])
            if class_i==0:
                label_idx = A
                unlabel_idx = B
            else:
                label_idx = np.concatenate((label_idx, A),axis=0)
                unlabel_idx = np.concatenate((unlabel_idx, B),axis=0)

        self.dataset1 = copy.deepcopy(dataset)
        self.dataset2 = copy.deepcopy(dataset)
        self.dataset1.samples = np.asarray(dataset.samples)[label_idx]
        self.dataset1.targets = np.asarray(dataset.targets)[label_idx]
        self.dataset2.samples = np.asarray(dataset.samples)[unlabel_idx]
        self.dataset2.targets = np.asarray(dataset.targets)[unlabel_idx]

        for i, (x,y) in enumerate(self.dataset1):
            if i == 0 :
                self.labeledData = x.unsqueeze(0)
                self.labeledLabel = torch.tensor([int(y)])
            else:
                self.labeledData    = torch.cat((self.labeledData, x.unsqueeze(0)),0)
                self.labeledLabel   = torch.cat((self.labeledLabel, torch.tensor([int(y)])),0)

    def get_all(self, dataset):
        self.unlabelset = dataset

class officeloader(object):
    def __init__(self, name_dataset, batch_size, label_ratio = 0.0,
                            shuffle=False, drop_last=False):
        # root dir (local pc or colab)
        root_dir = "data/office/%s/images" % name_dataset

        __datasets__ = ["amazon", "dslr", "webcam"]

        if name_dataset not in __datasets__:
            raise ValueError("must introduce one of the three datasets in office")

        # Ideally compute mean and std with get_mean_std_dataset.py
        # https://github.com/DenisDsh/PyTorch-Deep-CORAL/blob/master/data_loader.py

        # mean_std = {
        #     "amazon":{
        #         "mean":[0.7923, 0.7862, 0.7841],
        #         "std":[0.3149, 0.3174, 0.3193]
        #     },
        #     "dslr":{
        #         "mean":[0.4708, 0.4486, 0.4063],
        #         "std":[0.2039, 0.1920, 0.1996]
        #     },
        #     "webcam":{
        #         "mean":[0.6119, 0.6187, 0.6173],
        #         "std":[0.2506, 0.2555, 0.2577]
        #     }
        # }

        # # compose image transformations
        # data_transforms = transforms.Compose([
        #         transforms.Resize((224, 224)),
        #         # transforms.CenterCrop(224),
        #         # transforms.RandomSizedCrop(224),
        #         # transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=mean_std[name_dataset]["mean"],
        #                             std=mean_std[name_dataset]["std"])
        #     ])
        
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])



        self.nOutput = 31
        print('Number of output: ', self.nOutput)

        dataset = datasets.ImageFolder(root=root_dir,
                                    transform=data_transforms)

        #Number of initial dataset
        nSizeDataset    = len(dataset)
        indices         = torch.randperm(nSizeDataset)
        if label_ratio==0:
            unlabel_idx     = indices
            unlabel_sampler = SubsetRandomSampler(unlabel_idx)
            self.unlabel_dataset = DataLoader(dataset, batch_size=batch_size,
                                    # sampler=unlabel_sampler, 
                                    num_workers=0, shuffle=shuffle, drop_last=drop_last)
            self.nUnlabeledData = len(unlabel_sampler)
            print('Number of unlabeled data: ', self.nUnlabeledData)
            self.nBatch = len(self.unlabel_dataset)
            print('Number of unlabeled data batch: ', self.nBatch)
        else:
            nLabel          = round(nSizeDataset*label_ratio)
            import random
            labelIdx = []
            unlabelIdx = []
            for c in range(self.nOutput):
                # k = int(nLabel / self.nOutput) if c > (nLabel % self.nOutput) else (int(nLabel / self.nOutput) + 1)     # evenly distribute
                k = round(nLabel / self.nOutput)                                                                          # k samples for each class
                choices = random.sample([i for i,t in enumerate(dataset.targets) if t == c], k=k)
                [labelIdx.append(i) for i in choices]
                reverse = [i for i,t in enumerate(dataset.targets) if t == c]
                [reverse.remove(p) for p in choices]
                [unlabelIdx.append(j)  for j in reverse]
            # label_idx       = indices[:nLabel]
            # unlabel_idx     = indices[nLabel:]
            label_sampler = SubsetRandomSampler(labelIdx)
            unlabel_sampler = SubsetRandomSampler(unlabelIdx)

            # Create data loaders
            self.label_dataset = DataLoader(dataset, batch_size=batch_size,
                                    sampler=label_sampler, num_workers=0, shuffle=shuffle, drop_last=drop_last)
            self.unlabel_dataset = DataLoader(dataset, batch_size=batch_size,
                                    sampler=unlabel_sampler, num_workers=0, shuffle=shuffle, drop_last=drop_last)
                                    
            self.nLabeledData = len(label_sampler)
            print('Number of labeled data: ', self.nLabeledData)
            self.nUnlabeledData = len(unlabel_sampler)
            print('Number of unlabeled data: ', self.nUnlabeledData)
            self.nBatch = len(self.unlabel_dataset)
            print('Number of unlabeled data batch: ', self.nBatch)
    
    def get_labeled_distribution(self):
        import numpy as np
        try:
            for i, (_, y) in enumerate(self.label_dataset):
                if i == 0:
                    labeledLabel = y
                else:
                    labeledLabel = torch.cat((labeledLabel, y), 0)
            for i, (_, y) in enumerate(self.unlabel_dataset):
                if i == 0:
                    unlabeledLabel = y
                else:
                    unlabeledLabel = torch.cat((unlabeledLabel, y), 0)
            label1 = labeledLabel.numpy()
            label2 = unlabeledLabel.numpy()
            a = np.concatenate((label1,label2),axis=0)
            unique_elements, counts_elements = np.unique(label1, return_counts=True)
            print("Frequency of unique values of labeled source dataset array:")
            print(np.asarray((unique_elements, counts_elements)))
        except:
            for i, (_, y) in enumerate(self.unlabel_dataset):
                if i == 0:
                    unlabeledLabel = y
                else:
                    unlabeledLabel = torch.cat((unlabeledLabel, y), 0)
            a = unlabeledLabel.numpy()
        unique_elements, counts_elements = np.unique(a, return_counts=True)
        print("Frequency of unique values of the said array:")
        print(np.asarray((unique_elements, counts_elements)))

def office31_resnet50_loader(csv_file, batchsize, drop_last=False, shuffle=False):
    pf = pd.read_csv(csv_file)
    img = pf.iloc[:, :-1]
    label = pf.iloc[:, -1]
    dataset = data_utils.TensorDataset(torch.from_numpy(img.values).float(), torch.from_numpy(label.values).long())
    loader = DataLoader(dataset,batch_size=batchsize, drop_last=drop_last, shuffle=shuffle)
    return loader


# ================== Usage ================== #
class get_stl9(object):
    def __init__(self,label_rate=0.1):
        # Transforms object for testset with NO augmentation
        transform_no_aug = transforms.Compose([
                                                # transforms.ToPILImage(), 
                                                transforms.Resize(32), 
                                                transforms.ToTensor(), 
                                                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                                ])

        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context

        self.trainset = STL10(root='./data/stl10/', split='train', download=True, transform=transform_no_aug)
        self.testset = STL10(root='./data/stl10/', split='test', download=True, transform=transform_no_aug)
        
        classDict = {'airplane': 0, 'bird': 1, 'car': 2, 'cat': 3, 'deer': 4,
                    'dog': 5, 'horse': 6, 'monkey': 7, 'ship': 8, 'truck': 9}

        ''' 
        new class dict {'airplane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
                    'dog': 5, 'horse': 6, 'ship':7, 'truck': 8} 
        '''
        if label_rate == 0:
            self.get_all()
            del self.trainset, self.testset
        else:
            self.get_label_unlabel(label_rate)

    def get_label_unlabel(self,label_rate):
        delete_idx = (self.trainset.labels!=7)
        self.trainset.data = self.trainset.data[delete_idx]
        self.trainset.labels = self.trainset.labels[delete_idx]
        self.trainset.labels[self.trainset.labels==8] = 7
        self.trainset.labels[self.trainset.labels==9] = 8
        idx_bird = (self.trainset.labels==1).nonzero() # change to class 2
        idx_car  = (self.trainset.labels==2).nonzero() # change to class 1
        self.trainset.labels[idx_bird] = 2
        self.trainset.labels[idx_car] = 1

        delete_idx = (self.testset.labels!=7)
        self.testset.data = self.testset.data[delete_idx]
        self.testset.labels = self.testset.labels[delete_idx]
        self.testset.labels[self.testset.labels==8] = 7
        self.testset.labels[self.testset.labels==9] = 8
        idx_bird = (self.testset.labels==1).nonzero() # change to class 2
        idx_car  = (self.testset.labels==2).nonzero() # change to class 1
        self.testset.labels[idx_bird] = 2
        self.testset.labels[idx_car] = 1

        label_idx = unlabel_idx = []
        for class_i in range(9):
            idx = (self.trainset.labels==class_i).nonzero()[0]
            A, B = np.split(np.random.permutation(idx), [int(len(idx)*label_rate)])
            label_idx = np.concatenate((label_idx, A),axis=0)
            unlabel_idx = np.concatenate((unlabel_idx, B),axis=0)
        
        label_idx1 = unlabel_idx1 = []
        for class_i in range(9):
            idx = (self.testset.labels==class_i).nonzero()[0]
            A, B = np.split(np.random.permutation(idx), [int(len(idx)*label_rate)])
            label_idx1 = np.concatenate((label_idx1, A),axis=0)
            unlabel_idx1 = np.concatenate((unlabel_idx1, B),axis=0)

        data1 = np.concatenate((self.trainset.data[label_idx.astype(int)],self.testset.data[label_idx1.astype(int)]),axis=0)
        label1 = np.concatenate((self.trainset.labels[label_idx.astype(int)],self.testset.labels[label_idx1.astype(int)]),axis=0)
        data2 = np.concatenate((self.trainset.data[unlabel_idx.astype(int)],self.testset.data[unlabel_idx1.astype(int)]),axis=0)
        label2 = np.concatenate((self.trainset.labels[unlabel_idx.astype(int)],self.testset.labels[unlabel_idx1.astype(int)]),axis=0)

        self.trainset.data = data1
        self.trainset.labels = label1
        self.testset.data = data2
        self.testset.labels = label2

        unique_elements, counts_elements = np.unique(self.testset.labels, return_counts=True)
        print('Class distribution:\n',counts_elements)
        self.nOutput = len(unique_elements)
        
        for i, (x,y) in enumerate(self.trainset):
            if i == 0 :
                self.labeledData = x.unsqueeze(0)
                self.labeledLabel = torch.tensor([y])
            else:
                self.labeledData    = torch.cat((self.labeledData, x.unsqueeze(0)),0)
                self.labeledLabel   = torch.cat((self.labeledLabel, torch.tensor([y])),0)

    def get_all(self):
        self.unlabelset = ConcatDataset([self.trainset, self.testset])
        # data = np.concatenate((self.trainset.data, self.testset.data),0)
        # label = np.concatenate((self.trainset.labels, self.testset.labels),0)
        # self.unlabelset = TensorDataset(torch.tensor(data), torch.tensor(label).long())

class get_cifar9(object):
    def __init__(self,label_rate=0.1):
        # Transforms object for testset with NO augmentation
        transform_no_aug = transforms.Compose([
                                                # transforms.ToPILImage(), 
                                                transforms.Resize(32), 
                                                transforms.ToTensor(), 
                                                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                                ])

        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        self.trainset = CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform_no_aug)
        self.testset = CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform_no_aug)
        classDict = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
                    'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

        ''' 
        new class dict {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
                    'dog': 5, 'horse': 6, 'ship':7, 'truck': 8} 
        '''
        if label_rate == 0:
            self.get_all()
            del self.trainset, self.testset
        else:
            self.get_label_unlabel(label_rate)

    def get_all(self):
        self.unlabelset = ConcatDataset([self.trainset, self.testset])
        # data = np.concatenate((self.trainset.data, self.testset.data),0)
        # label = np.concatenate((self.trainset.targets, self.testset.targets),0)
        # self.unlabelset = TensorDataset(torch.tensor(np.reshape(data,(-1,3,32,32))), torch.tensor(label).long())
        
    def get_label_unlabel(self,label_rate):
        self.trainset.targets = np.asarray(self.trainset.targets)
        self.testset.targets = np.asarray(self.testset.targets)

        delete_idx = (self.trainset.targets!=6)
        self.trainset.data = self.trainset.data[delete_idx]
        self.trainset.targets = self.trainset.targets[delete_idx]
        self.trainset.targets[self.trainset.targets==7] = 6
        self.trainset.targets[self.trainset.targets==8] = 7
        self.trainset.targets[self.trainset.targets==9] = 8

        delete_idx = (self.testset.targets!=7)
        self.testset.data = self.testset.data[delete_idx]
        self.testset.targets = self.testset.targets[delete_idx]
        self.testset.targets[self.testset.targets==7] = 6
        self.testset.targets[self.testset.targets==8] = 7
        self.testset.targets[self.testset.targets==9] = 8

        label_idx = unlabel_idx = []
        for class_i in range(9):
            idx = (self.trainset.targets==class_i).nonzero()[0]
            A, B = np.split(np.random.permutation(idx), [int(len(idx)*label_rate)])
            label_idx = np.concatenate((label_idx, A),axis=0)
            unlabel_idx = np.concatenate((unlabel_idx, B),axis=0)
        
        label_idx1 = unlabel_idx1 = []
        for class_i in range(9):
            idx = (self.testset.targets==class_i).nonzero()[0]
            A, B = np.split(np.random.permutation(idx), [int(len(idx)*label_rate)])
            label_idx1 = np.concatenate((label_idx1, A),axis=0)
            unlabel_idx1 = np.concatenate((unlabel_idx1, B),axis=0)

        data1 = np.concatenate((self.trainset.data[label_idx.astype(int)],self.testset.data[label_idx1.astype(int)]),axis=0)
        label1 = np.concatenate((self.trainset.targets[label_idx.astype(int)],self.testset.targets[label_idx1.astype(int)]),axis=0)
        data2 = np.concatenate((self.trainset.data[unlabel_idx.astype(int)],self.testset.data[unlabel_idx1.astype(int)]),axis=0)
        label2 = np.concatenate((self.trainset.targets[unlabel_idx.astype(int)],self.testset.targets[unlabel_idx1.astype(int)]),axis=0)

        self.trainset.data = data1
        self.trainset.targets = label1
        self.testset.data = data2
        self.testset.targets = label2

        unique_elements, counts_elements = np.unique(self.testset.targets, return_counts=True)
        print('Class distribution:\n',counts_elements)
        self.nOutput = len(unique_elements)

        for i, (x,y) in enumerate(self.trainset):
            if i == 0 :
                self.labeledData = x.unsqueeze(0)
                self.labeledLabel = torch.tensor([y])
            else:
                self.labeledData    = torch.cat((self.labeledData, x.unsqueeze(0)),0)
                self.labeledLabel   = torch.cat((self.labeledLabel, torch.tensor([y])),0)