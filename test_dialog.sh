# CUDA_VISIBLE_DEVICES=7 python main.py -run_type predict -ckpt ./dialogue_t5_base_2.1/ckpt-epoch6 -output inference -batch_size 16
# CUDA_VISIBLE_DEVICES=7 python main.py -run_type predict -ckpt ./dialogue_t5_base_2.1/ckpt-epoch7 -output inference -batch_size 16
# CUDA_VISIBLE_DEVICES=7 python main.py -run_type predict -ckpt ./dialogue_t5_base_2.1/ckpt-epoch8 -output inference -batch_size 16
# CUDA_VISIBLE_DEVICES=7 python main.py -run_type predict -ckpt ./dialogue_t5_base_2.1/ckpt-epoch9 -output inference -batch_size 16
# CUDA_VISIBLE_DEVICES=7 python main.py -run_type predict -ckpt ./dialogue_t5_base_2.1/ckpt-epoch10 -output inference -batch_size 16
# CUDA_VISIBLE_DEVICES=7 python main.py -run_type predict -ckpt ./dialogue_t5_base_2.1/ckpt-epoch11 -output inference -batch_size 16
# CUDA_VISIBLE_DEVICES=7 python main.py -run_type predict -ckpt ./dialogue_t5_base_2.1/ckpt-epoch12 -output inference -batch_size 16
# CUDA_VISIBLE_DEVICES=7 python main.py -run_type predict -ckpt ./dialogue_t5_base_2.1/ckpt-epoch13 -output inference -batch_size 16
# CUDA_VISIBLE_DEVICES=7 python main.py -run_type predict -ckpt ./dialogue_t5_base_2.1/ckpt-epoch14 -output inference -batch_size 16
# CUDA_VISIBLE_DEVICES=7 python main.py -run_type predict -ckpt ./dialogue_t5_base_2.1/ckpt-epoch15 -output inference -batch_size 16
# CUDA_VISIBLE_DEVICES=7 python main.py -run_type predict -ckpt ./dialogue_t5_base_2.1/ckpt-epoch16 -output inference -batch_size 16
# CUDA_VISIBLE_DEVICES=7 python main.py -run_type predict -ckpt ./dialogue_t5_base_2.1/ckpt-epoch17 -output inference -batch_size 16
# CUDA_VISIBLE_DEVICES=7 python main.py -run_type predict -ckpt ./dialogue_t5_base_2.1/ckpt-epoch18 -output inference -batch_size 16
# CUDA_VISIBLE_DEVICES=7 python main.py -run_type predict -ckpt ./dialogue_t5_base_2.1/ckpt-epoch19 -output inference -batch_size 16
# CUDA_VISIBLE_DEVICES=7 python main.py -run_type predict -ckpt ./dialogue_t5_base_2.1/ckpt-epoch20 -output inference -batch_size 16
# CUDA_VISIBLE_DEVICES=5 python main.py -run_type predict -ckpt ./dialogue_t5_small/dialog_rl -output inference -batch_size 16
CUDA_VISIBLE_DEVICES=1 python main.py -run_type predict -ckpt ./dialogue_t5_small/dialog_rl -output inference.json -batch_size 16
CUDA_VISIBLE_DEVICES=1 python main.py -run_type predict -ckpt ./dialogue_t5_small/dialog_rl_epoch_ -output inference.json -batch_size 16
CUDA_VISIBLE_DEVICES=1 python main.py -run_type predict -ckpt ./dialogue_t5_small/dialog_rl_epoch_5 -output inference.json -batch_size 16
CUDA_VISIBLE_DEVICES=1 python main.py -run_type predict -ckpt ./dialogue_t5_small/dialog_rl_v4_epoch_3 -output inference.json -batch_size 16