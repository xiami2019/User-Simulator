CUDA_VISIBLE_DEVICES=2 python compute_all_scores.py -output_result_path dialog_t5_small_offline_ep11_onlineformat.json -config_dir dialogue_t5_small -eval_type online -gpt_score_singe_side
CUDA_VISIBLE_DEVICES=2 python compute_all_scores.py -output_result_path generate_example_test_t5_small_fixed.json -config_dir dialogue_t5_small -eval_type online -gpt_score_singe_side
CUDA_VISIBLE_DEVICES=2 python compute_all_scores.py -output_result_path generate_example_test_t5_small_rl_v1_fixed.json -config_dir dialogue_t5_small -eval_type online -gpt_score_singe_side
CUDA_VISIBLE_DEVICES=2 python compute_all_scores.py -output_result_path generate_example_test_t5_small_rl_v2_fixed.json -config_dir dialogue_t5_small -eval_type online -gpt_score_singe_side
CUDA_VISIBLE_DEVICES=2 python compute_all_scores.py -output_result_path generate_example_test_t5_small_rl_v3_fixed.json -config_dir dialogue_t5_small -eval_type online -gpt_score_singe_side
CUDA_VISIBLE_DEVICES=2 python compute_all_scores.py -output_result_path generate_example_test_t5_small_rl_v4_fixed.json -config_dir dialogue_t5_small -eval_type online -gpt_score_singe_side
CUDA_VISIBLE_DEVICES=2 python compute_all_scores.py -output_result_path generate_example_test_t5_small_rl_v5_fixed.json -config_dir dialogue_t5_small -eval_type online -gpt_score_singe_side