python -u main.py --device cuda --num_runs 5 --seed 0 --agent leopard      --labeled_rate 0.1   --batch 128  \
            --learning_rate 0.1 --dataset MnistUsps --epoch_kl 5 --epoch_dc 1   --epoch_clus_init 50 \
            --alpha_kl 1.0 --alpha_dc 0.1
python -u main.py --device cuda --num_runs 5 --seed 0 --agent adcn         --labeled_rate 0.1   --batch 128  \
            --learning_rate 0.1 --dataset MnistUsps --epoch_kl 1 --epoch_dc 1   --epoch_clus_init 50
python -u main.py --device cuda --num_runs 5 --seed 0 --agent ae-kmeans    --labeled_rate 0.1   --batch 128  \
            --learning_rate 0.1 --dataset MnistUsps --epochs 1                  --epoch_clus_init 50
python -u main.py --device cuda --num_runs 5 --seed 0 --agent dcn          --labeled_rate 0.1   --batch 128  \
            --learning_rate 0.1 --dataset MnistUsps --epochs 1                  --epoch_clus_init 50
python -u main.py --device cuda --num_runs 5 --seed 0 --agent dec          --labeled_rate 0.1   --batch 128  \
            --learning_rate 0.1 --dataset MnistUsps --epochs 1                    --epoch_clus_init 50
python -u main.py --device cuda --num_runs 5 --seed 0 --agent dann          --labeled_rate 0.1   --batch 128  \
            --learning_rate 0.1 --dataset MnistUsps --epochs 1                  --epoch_clus_init 50