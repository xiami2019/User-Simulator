# ppl

# gpt score
CUDA_VISIBLE_DEVICES=3 python train_lm.py -ckpt ./bart_score_gpt_lm_model_lr_1e_4/ckpt-epoch6 -run_type predict -ppl_level bart_score -batch_size 32