# Is MultiWOZ a Solved Task? An Interactive TOD Evaluation Framework with User Simulator
Accepted by the findings of EMNLP2022.
## Data Preprocess
Our experiments mainly focus on MultiWOZ 2.0.
You should unzip the data.zip at first:
```
cd data
unzip data.zip
```
and then run this script:
```
python preprocess.py -version 2.0
```

## Supervised Learning
Seq-to-seq training based on t5 models (simplified version of MTTOD).
### User Simulator Supvervised Training
```
python main.py -version 2.0 -agent_type us -run_type train -ururu -backbone t5-small -model_dir simulator_t5_small -epoch 20
```
### Dialogue System Supervised Training
```
python main.py -version 2.0 -agent_type ds -run_type train -ururu -backbone t5-small -model_dir dialogue_t5_small -epoch 20
```

### Interaction
Conduct interactions between a user simulator and a dialogue system (either SL-based models or RL-based models). Generate dialogue sessions based on user goals from test or dev set. This script can be used for different dialogue models(mttod, ubar, pptod).
```
python interact.py -simulator_path ./your_simulator_model_dir/checkpoint -dialog_sys_path ./your_dialogue_model_dir/your_checkpoint -model_name mttod -generate_results_path output.json
```

## Reinforcement Learning
### Using success rates as rewards
```
python interact.py -do_rl_training -seed 1998 -simulator_save_path simulator_rl -dialog_save_path dialog_rl
```
### Using success rates and sentence-score as rewards
```
python interact.py -do_rl_training -seed 1998 -simulator_save_path simulator_rl -dialog_save_path dialog_rl -use_gpt_score_as_reward -gpt_score_coef 0.1
```
### Using success rates and sessions-score as rewards
```
python interact.py -do_rl_training -seed 1998 -simulator_save_path simulator_rl -dialog_save_path dialog_rl -use_nsp_score_as_reward -nsp_coef 0.1
```

## Score Training
### Sentence Score Training
```
python train_lm.py -model_dir sentence_score_model -task ppl -ppl_level bart_score -backbone gpt2
```
### Sentence Score Evaluation
```
python train_lm.py -ckpt ./sentence_score_model/checkpoint -run_type predict -task ppl -ppl_level bart_score
```
### Session Score Training
```
python train_lm.py -model_dir session_score_model -task nsp -backbone bert-base-uncased
```
### Session Score Evaluation
```
python train_lm.py -ckpt ./session_score_model/checkpoint -run_type predict -task nsp
```

## Evaluation
### Traditional Evaluation
Computing Inform, Success and BLEU Score.
```
python main.py -run_type predict -predict_agent_type ds -ckpt ./dialogue_t5_small/checkpoint -output inference.json -batch_size 16
```
Computing BLEU Score(For evaluation of simulators).
```
python main.py -run_type predict -predict_agent_type us -ckpt ./simulator_t5_small/checkpoint -output inference.json -batch_size 16
```
### Interactive Evaluation
First generate dialogue by interactions between a user simulator and a dialogue system.
Then computing Inform, Success, Sentence-Score and Session-Score.  
```
python compute_all_scores.py -output_result_path output.json -config_dir dialogue_t5_small -eval_type online -lm_ckpt ./your_sentence_score_model/checkpoint -nsp_ckpt ./your_session_score_model/checkpoint
```
If the result is generated by traditional evaluation, you should convert its format to online format at first using:
```
python convert_offline_to_online_format.py -offline_path traditional_results.json -online_path traditional_results_online_format.json
```

## Different Models
### UBAR Training
```
python ubar.py --backbone distilgpt2 --run_type train --model_dir ubar_model 
```
### UBAR Evaluation
```
python ubar.py --ckpt ./ubar_model/checkpoint --run_type predict --pred_data_type test
```
### PPTOD Training
```
python pptod.py --backbone pptod_small --run_type train --model_dir pptod_model
```
### PPTOD Evaluation
```
python pptod.py --ckpt ./pptod_model/checkpoint --run_type predict --pred_data_type test
```
<!-- ### Galayx
We also incorporated Galaxy into our codes. However, our implementation did not achieve the performance reported in the original paper and thus we did not list the results of Galaxy in our paper.
#### Galaxy Training
```
python galaxy_finetune.py --save_dir galaxy_model_finetune
```
#### Galaxy Evluation
```
python galaxy_finetune.py --run_type predict
``` -->
