# CUDA_VISIBLE_DEVICES=1 python train_lm.py -model_dir gpt_lm_model_lr_1e_4 -learning_rate 1e-4
CUDA_VISIBLE_DEVICES=3 python train_lm.py -model_dir gpt_lm_model_lr_1e_4_user_side -learning_rate 1e-4 -backbone gpt2 -ppl_level bart_score -gpt_score_singe_side -agent usr