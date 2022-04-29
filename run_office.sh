# dslr <-> webcam
python -u main_office.py --device cuda --num_runs 5 --seed 0 --agent leopard          --labeled_rate 0.1  --batch 32  \
            --learning_rate 0.01 --source dslr --target webcam --backbone resnet34 \
            --epoch_clus_init 500 --epoch_kl 10 --epoch_dc 1 \
            --alpha_kl 1.0 --alpha_dc 0.1
python -u main_office.py --device cuda --num_runs 5 --seed 0 --agent leopard          --labeled_rate 0.1  --batch 32  \
            --learning_rate 0.01 --source webcam --target dslr --backbone resnet34 \
            --epoch_clus_init 500 --epoch_kl 10 --epoch_dc 1 \
            --alpha_kl 1.0 --alpha_dc 0.1

python -u main_office.py --device cuda --num_runs 5 --seed 0 --agent adcn          --labeled_rate 0.1  --batch 32  \
            --learning_rate 0.01 --source dslr --target webcam --backbone resnet34 \
            --epoch_clus_init 500 --epoch_kl 10 --epoch_dc 1 \
            --alpha_kl 1.0 --alpha_dc 0.1
python -u main_office.py --device cuda --num_runs 5 --seed 0 --agent adcn          --labeled_rate 0.1  --batch 32  \
            --learning_rate 0.01 --source webcam --target dslr --backbone resnet34 \
            --epoch_clus_init 500 --epoch_kl 10 --epoch_dc 1 \
            --alpha_kl 1.0 --alpha_dc 0.1

python -u main_office.py --device cuda --num_runs 5 --seed 0 --agent ae-kmeans          --labeled_rate 0.1  --batch 32  \
            --learning_rate 0.01 --source dslr --target webcam --backbone resnet34 \
            --epoch_clus_init 500 --epochs 10
python -u main_office.py --device cuda --num_runs 5 --seed 0 --agent ae-kmeans          --labeled_rate 0.1  --batch 32  \
            --learning_rate 0.01 --source webcam --target dslr --backbone resnet34 \
            --epoch_clus_init 500 --epochs 10

python -u main_office.py --device cuda --num_runs 5 --seed 0 --agent dcn          --labeled_rate 0.1  --batch 32  \
            --learning_rate 0.01 --source dslr --target webcam --backbone resnet34 \
            --epoch_clus_init 500 --epochs 10
python -u main_office.py --device cuda --num_runs 5 --seed 0 --agent dcn          --labeled_rate 0.1  --batch 32  \
            --learning_rate 0.01 --source webcam --target dslr --backbone resnet34 \
            --epoch_clus_init 500 --epochs 10

python -u main_office.py --device cuda --num_runs 5 --seed 0 --agent dec          --labeled_rate 0.1  --batch 32  \
            --learning_rate 0.01 --source dslr --target webcam --backbone resnet34 \
            --epoch_clus_init 500 --epochs 10
python -u main_office.py --device cuda --num_runs 5 --seed 0 --agent dec          --labeled_rate 0.1  --batch 32  \
            --learning_rate 0.01 --source webcam --target dslr --backbone resnet34 \
            --epoch_clus_init 500 --epochs 10

python -u main_office.py --device cuda --num_runs 5 --seed 0 --agent dann          --labeled_rate 0.1  --batch 32  \
            --learning_rate 0.01 --source dslr --target webcam --backbone resnet34 \
            --epoch_clus_init 500 --epochs 10
python -u main_office.py --device cuda --num_runs 5 --seed 0 --agent dann          --labeled_rate 0.1  --batch 32  \
            --learning_rate 0.01 --source webcam --target dslr --backbone resnet34 \
            --epoch_clus_init 500 --epochs 10