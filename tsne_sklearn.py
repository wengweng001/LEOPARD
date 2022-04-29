from sklearn.manifold import TSNE
from torchvision import transforms
from data_load import usps, mnist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import torch
from utilsADCN import generalExtractorMnistUsps

init = True

np.random.seed(0)
torch.manual_seed(0)

Mnist1 = mnist.MNIST('./data/mnist/', train=True, download=False,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            # transforms.Normalize((0.1307,), (0.3081,))
                        ]))
Mnist1.dataset_size=Mnist1.data.shape[0]
Mnist2 = mnist.MNIST('./data/mnist/', train=False, download=False,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            # transforms.Normalize((0.1307,), (0.3081,))
                        ]))
Mnist2.dataset_size=Mnist2.data.shape[0]
                        
#data USPS
Usps1 = usps.USPS_idx('./data/usps/', train=True, download=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                # transforms.Normalize((0.1307,), (0.3081,))
                            ]))
Usps2 = usps.USPS('./data/usps/', train=False, download=False,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            # transforms.Normalize((0.1307,), (0.3081,))
                        ]))
                        


dataset = generalExtractorMnistUsps(Mnist1, Mnist2)
# load model
if init:
    cnn = torch.load('init_cnn_UspsMnist.pt')
    ae  = torch.load('init_ae_UspsMnist.pt')
    data_1000   = dataset.unlabeledData[:1000].reshape(-1,784)
    labels_1000 = dataset.unlabeledLabel[:1000]
else:
    cnn = torch.load('trained_cnn_UspsMnist.pt')
    ae  = torch.load('trained_ae_UspsMnist.pt')
    data_1000   = ae[0].network(cnn(dataset.unlabeledData[:1000])).detach().cpu().numpy()
    labels_1000 = dataset.unlabeledLabel[:1000]

model = TSNE(n_components=2, random_state=0, perplexity=50, learning_rate=300)
# configuring the parameteres
# the number of components = 2
# default perplexity = 30
# default learning rate = 200
# default Maximum number of iterations for the optimization = 1000
tsne_data = model.fit_transform(data_1000)
# creating a new data frame which help us in ploting the result data
tsne_data = np.vstack((tsne_data.T, labels_1000)).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))
# Ploting the result of tsne
sb.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, "Dim_1", "Dim_2").add_legend()
plt.show()