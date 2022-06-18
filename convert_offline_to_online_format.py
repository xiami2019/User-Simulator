import argparse
from utils.utils import load_json, save_json

parser = argparse.ArgumentParser(description="Argument for preprocessing")
parser.add_argument('-offline_path', type=str, default=None, required=True)
parser.add_argument('-online_path', type=str, default=None, required=True)

args = parser.parse_args()

offline_path = args.offline_path
online_path = args.online_path

offline_data = load_json(offline_path)
online_data = []

for dial_id in offline_data:
    dial = {dial_id: {}, 'log': []}
    for turn in offline_data[dial_id]:
        single_turn = {}
        user = turn['user'].split(' ')[1:-1]
        user = ' '.join(user)
        resp = turn['resp_gen'].split(' ')[1:-1]
        resp = ' '.join(resp)
        single_turn['user'] = user
        single_turn['sys'] = resp
        single_turn['belief_states'] = turn['bspn_gen']
        single_turn["pointer"] = turn["pointer"]
        dial['log'].append(single_turn)
    online_data.append(dial)

save_json(online_data, online_path)

