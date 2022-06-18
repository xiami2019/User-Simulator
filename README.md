# Is MultiWOZ a Solved Task? An Interactive TOD Evaluation Framework with User Simulator

## Data Preprocess
(only MultiWOZ 2.0 now)
```
python preprocess.py -version 2.0
```

## Supervised Training
### Training
Useing seq2seq training based on t5 models.
#### User Simulator Supvervised Training
```
python main.py -version 2.0 -agent_type us -run_type train -ururu -backbone t5-small -model_dir simulator_t5_small -epoch 20
```
#### Dialogue System Supervised Training
```
python main.py -version 2.0 -agent_type ds -run_type train -ururu -backbone t5-small -model_dir dialogue_t5_small -epoch 20
```

### Interaction
Interaction between a user simulator and a dialogue system (either SL-based or RL-based). Generate new dialogue sessions based on user goals from test or dev set.

### RL Train
CUDA_VISIBLE_DEVICES=1 python interact.py -do_rl_training -seed 1998 -simulator_save_path simulator_rl_v5 -dialog_save_path dialog_rl_v5