import argparse
import random
import numpy as np
import torch
from Leopardplus_AM import run_leopard
from ADCN_AM import run_acdn
from DCN_AM import run_dcn
from AE_KMEANS_AM import run_aekmeans
from DEC_AM import run_dec
from DANN_AM import run_dann
from Leopardplus_mnist_usps import run_leopard_mu
from ADCN_Mnist_Usps import run_acdn_mu
from DCN_Mnist_Usps import run_dcn_mu
from AE_KMEANS_Mnist_Usps import run_aekmeans_mu
from DCN_Mnist_Usps import run_dcn_mu
from DEC_Mnist_Usps import run_dec_mu
from DANN_Mnist_Usps import run_dann_mu

import warnings
warnings.filterwarnings("ignore")

def main(args):
    print(args)
    # set up seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        args.device = torch.device('cpu')
    
    if args.dataset == 'AM':
        if args.agent == 'leopard':
            run_leopard(args)
        elif args.agent == 'adcn':
            run_acdn(args)
        elif args.agent == 'ae-kmeans':
            run_aekmeans(args)
        elif args.agent == 'dcn':
            run_dcn(args)
        elif args.agent == 'dec':
            run_dec(args)
        elif args.agent == 'dann':
            run_dann(args)
    elif 'Mnist' in args.dataset:
        if args.agent == 'leopard':
            run_leopard_mu(args)
        elif args.agent == 'adcn':
            run_acdn_mu(args)
        elif args.agent == 'ae-kmeans':
            run_aekmeans_mu(args)
        elif args.agent == 'dcn':
            run_dcn_mu(args)
        elif args.agent == 'dec':
            run_dec_mu(args)
        elif args.agent == 'dann':
            run_dann_mu(args)
        



if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Leopard++ Transfer Learning PyTorch")
    ########################General#########################
    parser.add_argument('--num_runs', dest='num_runs', default=5, type=int,
                        help='Number of runs (default: %(default)s)')
    parser.add_argument('--seed', dest='seed', default=0, type=int,
                        help='Random seed')
    parser.add_argument('--device', dest='device', default='cpu', choices=['cpu', 'cuda'],
                        help='Device (default: %(default)s)')

    ########################Agent###########################
    parser.add_argument('--agent', dest='agent', default='leopard',
                        choices=['leopard', 'adcn', 'ae-kmeans', 'dcn', 'dec', 'dann'],
                        help='Agent selection  (default: %(default)s)')

    ########################Optimizer#######################
    parser.add_argument('--optimizer', dest='optimizer', default='SGD', choices=['SGD', 'Adam'],
                        help='Optimizer (default: %(default)s)')
    parser.add_argument('--learning_rate', dest='learning_rate', default=0.01,
                        type=float,
                        help='Learning_rate (default: %(default)s)')
    parser.add_argument('--batch', dest='batch', default=16,
                        type=int,
                        help='Batch size (default: %(default)s)')

    ########################Data############################
    parser.add_argument('--dataset', type=str, default='AM', help="AM(default)|MnistUsps|UspsMnist")
    parser.add_argument('--labeled_rate', type=float, default=0.1, help="labeled percentage for source dataset")

    ########################LeopardPlus&ADCN################
    parser.add_argument('--epoch_kl', dest='epoch_kl', default=1,
                        type=int,
                        help='The number of epochs during prequential (CNN, KL and Auto Encoder) for leopard and (CNN and Auto Encoder) for adcn. (default: %(default)s)')
    parser.add_argument('--epoch_dc', dest='epoch_dc', default=1,
                        type=int,
                        help='The number of epochs during prequential (Domain adaptation) for leopard and (KL and L1) for adcn. (default: %(default)s)')
    parser.add_argument('--epoch_clus_init', dest='epoch_clus_init', default=100,
                        type=int,
                        help='The number of epoch during cluster initialization for leopard and adcn. (default: %(default)s)')
    parser.add_argument('--alpha_kl', dest='alpha_kl', default=1.0,
                        type=float,
                        help='The coefficient for regularization in Kullback–Leibler divergence. (default: %(default)s)')
    parser.add_argument('--alpha_dc', dest='alpha_dc', default=0.1,
                        type=float,
                        help='The coefficient for regularization for domain adaptation. (default: %(default)s)')

    ########################AE_kmeans#######################
    parser.add_argument('--epochs', dest='epochs', default=1,
                        type=int,
                        help='The number of epochs during training. (default: %(default)s)')

    args = parser.parse_args()
    if args.device == 'cuda':
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device('cpu')
    main(args)