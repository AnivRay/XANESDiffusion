import os

qm9Command = "python main_qm9.py --no_wandb --n_epochs 3000 --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 32 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999 --train_diffusion --trainable_ae --latent_nf 1 --exp_name geoldm_qm9"

drugsCommand = "CUDA_VISIBLE_DEVICES=0 python main_geom_drugs.py --no_wandb --n_epochs 3000 --n_stability_samples 500 --diffusion_noise_schedule polynomial_2 --diffusion_steps 1000 --diffusion_noise_precision 1e-5 --diffusion_loss_type l2 --batch_size 32 --nf 256 --n_layers 4 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 1 --ema_decay 0.9999 --normalization_factor 1 --model egnn_dynamics --visualize_every_batch 10000 --train_diffusion --trainable_ae --latent_nf 2 --exp_name geoldm_drugs"

drugsTestCommand = "CUDA_VISIBLE_DEVICES=0 python xanes_test.py --model_path outputs/geoldm_drugs"

xanesCondCommand = "CUDA_VISIBLE_DEVICES=0 python main_geom_drugs.py --no_wandb --conditioning xanes --sin_embedding False --dequantization deterministic --n_epochs 3000 --n_stability_samples 500 --diffusion_noise_schedule polynomial_2 --diffusion_steps 1000 --diffusion_noise_precision 1e-5 --diffusion_loss_type l2 --batch_size 32 --nf 256 --n_layers 4 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 1 --ema_decay 0.9999 --normalization_factor 1 --model egnn_dynamics --train_diffusion --trainable_ae --latent_nf 2 --exp_name xanes_cond"

os.system(xanesCondCommand)
