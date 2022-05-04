import json
from utils.utils import load_json, save_json

version = "2.0"

if version == "2.0":
    train_data_path = './data/MultiWOZ_2.0/processed/train_data.json'
    dev_data_path = './data/MultiWOZ_2.0/processed/dev_data.json'
    test_data_path = './data/MultiWOZ_2.0/processed/test_data.json'

    new_train_data_path = './data/MultiWOZ_2.0/processed/bart_score_train_data.json'
    new_dev_data_path = './data/MultiWOZ_2.0/processed/bart_score_dev_data.json'
    new_test_data_path = './data/MultiWOZ_2.0/processed/bart_score_test_data.json'
else:
    train_data_path = './data/MultiWOZ_2.1/processed/train_data.json'
    dev_data_path = './data/MultiWOZ_2.1/processed/dev_data.json'
    test_data_path = './data/MultiWOZ_2.1/processed/test_data.json'

    new_train_data_path = './data/MultiWOZ_2.1/processed/bart_score_train_data.json'
    new_dev_data_path = './data/MultiWOZ_2.1/processed/bart_score_dev_data.json'
    new_test_data_path = './data/MultiWOZ_2.1/processed/bart_score_test_data.json'

old_train_data = load_json(train_data_path)
old_dev_data = load_json(dev_data_path)
old_test_data = load_json(test_data_path)
train_data = []
dev_data = []
test_data = []

for dial_id in old_train_data:
    dial_history = 'session starts.'
    for turn in old_train_data[dial_id]['log']:
        train_data.append({'text': dial_history, 'summary': turn['user']})
        dial_history += ' ' + turn['user']
        train_data.append({'text': dial_history, 'summary': turn['resp']})
        dial_history += ' ' + turn['resp']

for dial_id in old_dev_data:
    dial_history = 'session starts.'
    for turn in old_dev_data[dial_id]['log']:
        dev_data.append({'text': dial_history, 'summary': turn['user']})
        dial_history += ' ' + turn['user']
        dev_data.append({'text': dial_history, 'summary': turn['resp']})
        dial_history += ' ' + turn['resp']

for dial_id in old_test_data:
    dial_history = 'session starts.'
    for turn in old_test_data[dial_id]['log']:
        test_data.append({'text': dial_history, 'summary': turn['user']})
        dial_history += ' ' + turn['user']
        test_data.append({'text': dial_history, 'summary': turn['resp']})
        dial_history += ' ' + turn['resp']

with open(new_train_data_path, 'w', encoding='utf-8') as f:
    for i in train_data:
        jsonstr = json.dumps(i)
        f.write(jsonstr)
        f.write('\n')

with open(new_dev_data_path, 'w', encoding='utf-8') as f:
    for i in dev_data:
        jsonstr = json.dumps(i)
        f.write(jsonstr)
        f.write('\n')

with open(new_test_data_path, 'w', encoding='utf-8') as f:
    for i in test_data:
        jsonstr = json.dumps(i)
        f.write(jsonstr)
        f.write('\n')


# save_json(train_data, new_train_data_path)
# save_json(dev_data, new_dev_data_path)
# save_json(test_data, new_test_data_path)







