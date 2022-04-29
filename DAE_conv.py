import torch
import torch.nn as nn
import torch.nn.functional as F
import backbone

from utilsADCN import ReverseLayerF


class domain_classifier_3layer(nn.Module):
    def __init__(self,no_input, no_hidden):
        super(domain_classifier_3layer, self).__init__()
        self.linear1 = nn.Linear(no_input, no_hidden, bias=True)
        self.linear2 = nn.Linear(no_hidden, no_hidden, bias=True)
        self.activation = nn.ReLU(inplace=True)
        self.linear3 = nn.Linear(no_hidden, 2)
        self.dropput = nn.Dropout()
        # self.logsoftmax = nn.LogSoftmax(dim=1)

        nn.init.xavier_uniform_(self.linear1.weight)
        self.linear1.bias.data.zero_()
        nn.init.xavier_uniform_(self.linear2.weight)
        self.linear2.bias.data.zero_()
        nn.init.xavier_uniform_(self.linear3.weight)
        self.linear3.bias.data.zero_()

    def forward(self, x, alpha=1.0):
        reverse_feature = ReverseLayerF.apply(x, alpha)
        x = self.linear1(reverse_feature)
        x = self.activation(x)
        x = self.dropput(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.dropput(x)
        x = self.linear3(x)
        # x = self.logsoftmax(x)
        return x

class Decoder(nn.Module):
    def __init__(self, ninput):
        super(Decoder,self).__init__()
        self.dfc2 = nn.Linear(ninput, 2048)
        self.bn2 = nn.BatchNorm2d(2048)
        self.dfc1 = nn.Linear(2048,256 * 3 * 3)
        self.bn1 = nn.BatchNorm2d(256*3*3)
        self.upsample1=nn.Upsample(scale_factor=2)
        self.dconv2 = nn.ConvTranspose2d(256, 64, 7, stride = 4)
        self.dconv1 = nn.ConvTranspose2d(64, 3, 14, stride = 4, padding = 1)

        self.init_weight()
    
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self,x):#,i1,i2,i3):
        x = self.dfc2(x)
        x = F.relu(x)
        x = self.dfc1(x)
        x = F.relu(x)
        x = x.view(x.size(0),256,3,3)
        x = self.upsample1(x)
        x = self.dconv2(x)
        x = F.relu(x)
        x = self.upsample1(x)
        x = self.dconv1(x)
        x = torch.tanh(x)
        return x

class Decoder32(nn.Module):
    def __init__(self, ninput):
        super(Decoder32,self).__init__()
        self.dfc1 = nn.Linear(ninput, 128 * 3 * 3)
        self.bn1 = nn.BatchNorm2d(128 * 3 * 3)
        self.upsample1=nn.Upsample(scale_factor=2)
        self.dconv2 = nn.ConvTranspose2d(128, 64, 5, stride = 2)
        self.dconv1 = nn.ConvTranspose2d(64, 3, 6, stride = 2, padding = 1)
        self.sigmoid = nn.Sigmoid()

        self.init_weight()
    
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)


    def forward(self,x):#,i1,i2,i3):
        x = self.dfc1(x)
        x = F.relu(x)
        x = x.view(x.size(0),128,3,3)
        x = self.upsample1(x)
        x = self.dconv2(x)
        x = F.relu(x)
        x = self.dconv1(x)
        x = self.sigmoid(x)
        return x

class ConvAe(nn.Module):
    def __init__(self, base_net='resnet34'):
        super(ConvAe, self).__init__()
        ## encoder layers ##
        self.encoder = backbone.network_dict[base_net]()

        # if base_net=='resnet18' or base_net=='resnet34' or base_net='resnet50':
        if 'resnet' in base_net:
            bottleneck=1000
        elif base_net=='vgg16' or base_net=='alexnet':
            bottleneck=2048
            
        self.bottleneck = nn.Linear(self.encoder.output_num(), bottleneck)
        self.bottleneck.weight.data.normal_(0, 0.005)
        self.bottleneck.bias.data.fill_(0.1)
        self.__features = bottleneck

        # self.t_conv1    = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.decoder = Decoder(self.encoder.output_num())
        self.biasDecoder1 = nn.Parameter(torch.zeros(self.encoder.output_num()))

    def forward(self, x, mode=1):
        # mode 1. encode, 2. decode
        if mode == 1:
            ## encode ##
            x = self.encoder(x)
            x = self.bottleneck(x)

        if mode == 2:
            ## decode ##
            x = F.linear(x, self.bottleneck.weight.t()) + self.biasDecoder1
            x = self.decoder(x)

        return x

    def feature_num(self):
        return self.__features

class ConvAe32(nn.Module):
    def __init__(self, base_net='resnet34'):
        super(ConvAe32, self).__init__()
        ## encoder layers ##
        self.encoder = backbone.network_dict[base_net]()

        # if base_net=='resnet18' or base_net=='resnet34' or base_net='resnet50':
        if 'resnet' in base_net:
            bottleneck=1000
        elif base_net=='vgg16' or base_net=='alexnet':
            bottleneck=2048
            
        self.bottleneck = nn.Linear(self.encoder.output_num(), bottleneck)
        self.bottleneck.weight.data.normal_(0, 0.005)
        self.bottleneck.bias.data.fill_(0.1)
        self.__features = bottleneck

        # self.t_conv1    = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.decoder = Decoder32(self.encoder.output_num())
        self.biasDecoder1 = nn.Parameter(torch.zeros(self.encoder.output_num()))

    def forward(self, x, mode=1):
        # mode 1. encode, 2. decode
        if mode == 1:
            ## encode ##
            x = self.encoder(x)
            x = self.bottleneck(x)

        if mode == 2:
            ## decode ##
            x = F.linear(x, self.bottleneck.weight.t()) + self.biasDecoder1
            x = self.decoder(x)

        return x

    def feature_num(self):
        return self.__features

class ConvDAE(nn.Module):
    def __init__(self, base_net='resnet34'):
        super(ConvDAE, self).__init__()
        ## encoder layers ##
        self.encoder = backbone.network_dict[base_net]()

        # if base_net=='resnet18' or base_net=='resnet34' or base_net='resnet50':
        if 'resnet' in base_net:
            bottleneck=1000
        elif base_net=='vgg16' or base_net=='alexnet':
            bottleneck=2048
            
        self.bottleneck = nn.Linear(self.encoder.output_num(), bottleneck)
        self.bottleneck.weight.data.normal_(0, 0.005)
        self.bottleneck.bias.data.fill_(0.1)
        self.__features = bottleneck

        # self.t_conv1    = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.decoder = Decoder(self.encoder.output_num())
        self.biasDecoder1 = nn.Parameter(torch.zeros(self.encoder.output_num()))

        ## fully connected layer ##
        # encoder
        self.linear = nn.Linear(bottleneck, 500,  bias=True)
        self.relu = nn.ReLU(inplace=True)
        nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.zero_()
        # decoder
        self.biasDecoder = nn.Parameter(torch.zeros(bottleneck))

    def forward(self, x, mode=1):
        # mode 1. encode, 2. decode
        if mode == 1:
            ## encode ##
            x = self.encoder(x)
            x = self.bottleneck(x)

            # add fully connected layer
            x = self.linear(x)
            x = self.relu(x)

        if mode == 2:
            ## decode ##
            # add fully connected layer
            x = F.linear(x, self.linear.weight.t()) + self.biasDecoder
            x = self.relu(x)                          # reconstructed input for end-to-end cnn

            x = F.linear(x, self.bottleneck.weight.t()) + self.biasDecoder1
            x = self.decoder(x)

        return x

    def feature_num(self):
        return self.__features