CUDA_VISIBLE_DEVICES=0 python compute_all_scores.py -output_result_path dialog_t5_small_offline_ep11_onlineformat.json -config_dir dialogue_t5_small -eval_type online
CUDA_VISIBLE_DEVICES=0 python compute_all_scores.py -output_result_path generate_example_test_t5_small.json -config_dir dialogue_t5_small -eval_type online
CUDA_VISIBLE_DEVICES=0 python compute_all_scores.py -output_result_path generate_example_test_t5_small_rl_v1.json -config_dir dialogue_t5_small -eval_type online
CUDA_VISIBLE_DEVICES=0 python compute_all_scores.py -output_result_path generate_example_test_t5_small_rl_v2.json -config_dir dialogue_t5_small -eval_type online
CUDA_VISIBLE_DEVICES=0 python compute_all_scores.py -output_result_path generate_example_test_t5_small_rl_v3.json -config_dir dialogue_t5_small -eval_type online
CUDA_VISIBLE_DEVICES=0 python compute_all_scores.py -output_result_path generate_example_test_t5_small_rl_v4.json -config_dir dialogue_t5_small -eval_type online
CUDA_VISIBLE_DEVICES=0 python compute_all_scores.py -output_result_path generate_example_test_t5_small_rl_nsp_gptscore_v4.json -config_dir dialogue_t5_small -eval_type online