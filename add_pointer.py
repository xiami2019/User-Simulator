from utils.utils import load_json, save_json

gen_examples = load_json('generate_example_2.json')
outputs = load_json('./dialogue_t5_base_predict_db/ckpt-epoch10/inference')

for dial in gen_examples:
    for key in dial.keys():
            if key.endswith('json'):
                dial_id = key
    
    for index, single_turn in enumerate(dial['log']):
        single_turn['pointer'] = outputs[dial_id][index]['pointer']

save_json(gen_examples, 'generate_example_test.json')