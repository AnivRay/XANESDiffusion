import os

# qm9Command = "python main_qm9.py --no_wandb --n_epochs 3000 --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 32 --nf 256 --n_layers 9 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 20 --ema_decay 0.9999 --train_diffusion --trainable_ae --latent_nf 1 --exp_name geoldm_qm9"
# drugsCommand = "CUDA_VISIBLE_DEVICES=0 python main_geom_drugs.py --no_wandb --n_epochs 3000 --n_stability_samples 500 --diffusion_noise_schedule polynomial_2 --diffusion_steps 1000 --diffusion_noise_precision 1e-5 --diffusion_loss_type l2 --batch_size 32 --nf 256 --n_layers 4 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 1 --ema_decay 0.9999 --normalization_factor 1 --model egnn_dynamics --visualize_every_batch 10000 --train_diffusion --trainable_ae --latent_nf 2 --exp_name geoldm_drugs"
# drugsTestCommand = "CUDA_VISIBLE_DEVICES=0 python xanes_test.py --model_path outputs/geoldm_drugs"

# xanesLinearCommand = "CUDA_VISIBLE_DEVICES=0 python linear_baseline_trainer.py --conditioning xanes --n_epochs 3000  --batch_size 64 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 1 --normalization_factor 1 --exp_name ti_xanes_large_linear"
# xanesLinearGenCommand = "CUDA_VISIBLE_DEVICES=0 python eval_linear_baseline.py"

# combinedCommand = "CUDA_VISIBLE_DEVICES=0 python eval_combined_baseline.py --generators_path outputs/fe_xanes_cond2_larger_bs_longer"
# numAtomCommand = "CUDA_VISIBLE_DEVICES=0 python num_atoms_trainer.py --no_wandb --n_epochs 200 --batch_size 64 --lr 5e-5 --exp_name num_atoms_classifier"
# CNClassifierCommand = "python CN_classifier.py --metal Fe"

xanesCondCommand = "CUDA_VISIBLE_DEVICES=0 python main_geom_drugs.py --conditioning xanes --sin_embedding False --dequantization deterministic --n_epochs 3000 --n_stability_samples 500 --diffusion_noise_schedule polynomial_2 --diffusion_steps 1000 --diffusion_noise_precision 1e-5 --diffusion_loss_type l2 --batch_size 128 --nf 512 --n_layers 8 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 1 --ema_decay 0.9999 --normalization_factor 1 --model egnn_dynamics --train_diffusion --trainable_ae --latent_nf 2 --exp_name ti_xanes_cond"
xanesUncondCommand = "CUDA_VISIBLE_DEVICES=1 python main_geom_drugs.py --sin_embedding False --dequantization deterministic --n_epochs 3000 --n_stability_samples 500 --diffusion_noise_schedule polynomial_2 --diffusion_steps 1000 --diffusion_noise_precision 1e-5 --diffusion_loss_type l2 --batch_size 128 --nf 512 --n_layers 8 --lr 1e-4 --normalize_factors [1,4,10] --test_epochs 1 --ema_decay 0.9999 --normalization_factor 1 --model egnn_dynamics --train_diffusion --trainable_ae --latent_nf 2 --exp_name ti_xanes_uncond"

xanesCondGenCommand = "CUDA_VISIBLE_DEVICES=5 python eval_conditional_xanes.py --generators_path outputs/mn_xanes_cond"
kNNCommand = "CUDA_VISIBLE_DEVICES=0 python eval_knn_baseline.py --no-cuda --generators_path outputs/mn_xanes_cond"

pdfAnalysisCommand = "python pdf_knn_analysis.py --no-cuda --generators_path outputs/fe_xanes_cond_final"


os.system(xanesUncondCommand)
