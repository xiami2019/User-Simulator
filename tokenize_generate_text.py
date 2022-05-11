from utils.utils import load_json, save_json
from tqdm import tqdm

text_path = 'generate_example_test_t5_small_rl_v5.json'
data = load_json(text_path)

punctuation = [',', '.', '?', '!']

for dial in tqdm(data):
    for turn in dial['log']:
        # print(turn['user'])
        new_user_ids = []
        user_ids = turn['user'].split()
        for token in user_ids:
            if token[-1] in punctuation:
                new_user_ids.append(token[:-1])
                new_user_ids.append(token[-1])
            else:
                new_user_ids.append(token)
        turn['user'] = ' '.join(new_user_ids)

        new_sys_ids = []
        sys_ids = turn['sys'].split()
        for token in sys_ids:
            if token[-1] in punctuation:
                new_sys_ids.append(token[:-1])
                new_sys_ids.append(token[-1])
            else:
                new_sys_ids.append(token)
        turn['sys'] = ' '.join(new_sys_ids)

save_json(data, text_path[:-5] + '_fixed.json')