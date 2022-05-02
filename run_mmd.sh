# python -u ablation_study.py --device cuda --num_runs 5 --seed 0 --agent leopard      --labeled_rate 0.1   --batch 32  \
#             --dataset AM --epoch_kl 1 --epoch_dc 1   --epoch_clus_init 100 --alpha_kl 1.0 --alpha_dc 0.1 --mmd --mmd_kernel rbf

# python -u ablation_study.py --device cuda --num_runs 5 --seed 0 --agent leopard      --labeled_rate 0.1   --batch 32  \
#             --dataset AM --epoch_kl 1 --epoch_dc 1   --epoch_clus_init 100 --alpha_kl 1.0 --alpha_dc 0.1 --mmd --mmd_kernel multiscale
        
# python -u ablation_study.py --device cuda --num_runs 5 --seed 0 --agent leopard      --labeled_rate 0.1   --batch 32  \
#             --dataset AM --epoch_kl 1 --epoch_dc 1   --epoch_clus_init 100 --alpha_kl 1.0 --alpha_dc 0.1 --mmd --mmd_kernel gaussian

        
python -u ablation_study.py --device cuda --num_runs 5 --seed 0 --agent leopard      --labeled_rate 0.1   --batch 32  \
            --dataset AM --epoch_kl 1 --epoch_dc 1   --epoch_clus_init 100 --alpha_kl 1.0 --alpha_dc 0.1 --mmd_d
