# LEOPARD
This is a PyTorch implementation of the transfer learning experiments described in "Autonomous Cross Domain Adaptation under Extreme Label Scarcity".


## Requirements
The current version of the code has been tested with the following configuration:
- python 3.6
- pytorch 1.9.0
- torchvision 0.10.0

## Run
### Individual experment can be run with `main.py`
Main options are:
--agent: Select agent (algorithm)? (leopard|adcn|ae-kmeans|dcn|dec)

To run specific methods, take **Amazon Review** as an example, you can use the following:
- Leopard++ : 
```
./main.py --device cuda --num_runs 5 --seed 0 --agent leopard   --labeled_rate 0.1  --batch 32 --dataset AM --epoch_kl 1 --epoch_dc 1 --epoch_clus_init 100 --alpha_kl 1.0 --alpha_dc 0.1
```
- ADCN:
```
./main.py --device cuda --num_runs 5 --seed 0 --agent adcn      --labeled_rate 0.1  --batch 32 --dataset AM --epoch_kl 1 --epoch_dc 1   --epoch_clus_init 100
```
- AE-Kmeans:
```
./main.py --device cuda --num_runs 5 --seed 0 --agent ae-kmeans --labeled_rate 0.1  --batch 32 --dataset AM --epochs 1 --epoch_clus_init 100
```
- DCN:
```
./main.py --device cuda --num_runs 5 --seed 0 --agent dcn       --labeled_rate 0.1  --batch 32 --dataset AM --epochs 1 --epoch_clus_init 100
```
- DEC:
```
./main.py --device cuda --num_runs 5 --seed 0 --agent dec       --labeled_rate 0.1  --batch 32 --dataset AM --epochs 1 --epoch_clus_init 100
```



### Experiments scripts
- Amazon Review experiments: `run_am.sh`
- MNIST to USPS: `run_mnist2usps.sh`
- USPS to MNIST: `run_usps2mnist.sh`
- Office-31: `run_office.sh`
- partial ablation study result: `run_ablation.sh` \
Experiments result in the paper can be viewed under `csv` folder.

