python -u main.py --device cuda --num_runs 1 --seed 0 --agent leopard      --labeled_rate 0.1   --batch 32  \
            --learning_rate 0.01 --dataset UspsMnist --epoch_kl 1 --epoch_dc 1   --epoch_clus_init 5 --alpha_kl 1.0 --alpha_dc 0.1
python -u main.py --device cuda --num_runs 5 --seed 0 --agent adcn         --labeled_rate 0.1   --batch 32  \
            --learning_rate 0.1 --dataset UspsMnist --epoch_kl 1 --epoch_dc 1   --epoch_clus_init 5
python -u main.py --device cuda --num_runs 5 --seed 0 --agent ae-kmeans    --labeled_rate 0.1   --batch 32  \
            --learning_rate 0.1 --dataset UspsMnist --epochs 1                  --epoch_clus_init 50
python -u main.py --device cuda --num_runs 5 --seed 0 --agent dcn          --labeled_rate 0.1   --batch 32  \
            --learning_rate 0.1 --dataset UspsMnist --epochs 1                  --epoch_clus_init 50
python -u main.py --device cuda --num_runs 5 --seed 0 --agent dec          --labeled_rate 0.1   --batch 32  \
            --learning_rate 0.1 --dataset UspsMnist --epochs 1                  --epoch_clus_init 50
python -u main.py --device cuda --num_runs 5 --seed 0 --agent dann          --labeled_rate 0.1   --batch 32  \
            --learning_rate 0.1 --dataset UspsMnist --epochs 1                  --epoch_clus_init 50
