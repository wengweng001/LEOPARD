import torch.nn as nn
import backbone
from utilsADCN import ReverseLayerF

class DANN_office(nn.Module):
    def __init__(self, base_net):
        super(DANN_office, self).__init__()
        
        self.feature = backbone.network_dict[base_net]()
        
        # if base_net=='resnet18' or base_net=='resnet34' or base_net='resnet50':
        if 'resnet' in base_net:
            bottleneck=1000
        elif base_net=='vgg16' or base_net=='alexnet':
            bottleneck=2048
        
        self.bottleneck = nn.Sequential()
        self.bottleneck.add_module('bottleneck',nn.Linear(self.feature.output_num(), bottleneck))
        self.bottleneck.add_module('relu',nn.ReLU(inplace=True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(bottleneck, 500))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_fc2', nn.Linear(500, 31))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(bottleneck, 500))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(500, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))


    def forward(self, input_data, alpha):
        feature = self.feature(input_data)
        feature = self.bottleneck(feature)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output