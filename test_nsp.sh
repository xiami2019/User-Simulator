CUDA_VISIBLE_DEVICES=4 python train_lm.py -model_dir bert_nsp_model_lr_1e_4_1 -learning_rate 1e-4 -task nsp -backbone bert-base-uncased -batch_size 32