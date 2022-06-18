from utils.utils import load_json, save_json

file_name = './data/MultiWOZ_2.0/processed/data_for_galaxy.json'
galaxy_data = load_json(file_name)

dev = './data/MultiWOZ_2.0/processed/dev_data.json'
test = './data/MultiWOZ_2.0/processed/test_data.json'
train = './data/MultiWOZ_2.0/processed/train_data.json'

dev_new = './data/MultiWOZ_2.0/processed/dev_data_galaxy.json'
test_new = './data/MultiWOZ_2.0/processed/test_data_galaxy.json'
train_new = './data/MultiWOZ_2.0/processed/train_data_galaxy.json'

dev_data = load_json(dev)
test_data = load_json(test)
train_data = load_json(train)

for dial_id in dev_data:
    for t, turn in enumerate(dev_data[dial_id]['log']):
        turn['unified_act'] = galaxy_data[dial_id[:-5]]['log'][t]['unified_act']

for dial_id in test_data:
    for t, turn in enumerate(test_data[dial_id]['log']):
        turn['unified_act'] = galaxy_data[dial_id[:-5]]['log'][t]['unified_act']

for dial_id in train_data:
    for t, turn in enumerate(train_data[dial_id]['log']):
        turn['unified_act'] = galaxy_data[dial_id[:-5]]['log'][t]['unified_act']
        

save_json(dev_data, dev_new)
save_json(test_data, test_new)
save_json(train_data, train_new)