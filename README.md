# User Simulator

## Supervised Training
### Data preprocess
(only MultiWOZ 2.0 now)
```
python preprocess.py
```

### Training
#### User Simulator Supvervised Training
```
python main.py -version 2.0 -agent_type us -run_type train -model_dir simulator_t5_base -log_frequency 300 -epoch 20
```
#### Dialogue System Supervised Training
```
python main.py -version 2.0 -agent_type ds -run_type train -model_dir dialogue_t5_base -log_frequency 300 -epoch 20
```

### Next Step
Interaction between us and ds.
Generating new dialogue sessions.

### RL Train
CUDA_VISIBLE_DEVICES=1 python interact.py -do_rl_training -seed 1998 -simulator_save_path simulator_rl_v5 -dialog_save_path dialog_rl_v5