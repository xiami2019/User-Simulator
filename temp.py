from utils.utils import load_json, save_json

file1 = 'dialog_t5_small_offline_ep11_onlineformat.json'
file2 = 'test_data.json'
data1 = load_json(file1)
data2 = load_json(file2)

for dial in data1:
    for key in dial.keys():
        if key.endswith('.json'):
            dial_id = key
            break
    
    for t, turn in enumerate(dial['log']):
        turn['sys'] = data2[dial_id]['log'][t]['resp']
        turn["belief_states"] = '<bos_belief> ' + data2[dial_id]['log'][t]['constraint'] + ' <eos_belief>'

save_json(data1, 'test_data_online_format.json')