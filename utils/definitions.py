ALL_DOMAINS = ["attraction", "hotel", "restaurant", "taxi", "train", "hospital", "police"]

NORMALIZE_SLOT_NAMES = {
    "car type": "car",
    "entrance fee": "price",
    "duration": "time",
    "leaveat": 'leave',
    'arriveby': 'arrive',
    'trainid': 'id'
}

REQUESTABLE_SLOTS = {
    "taxi": ["car", "phone"],
    "police": ["postcode", "address", "phone"],
    "hospital": ["address", "phone", "postcode"],
    "hotel": ["address", "postcode", "internet", "phone", "parking",
              "type", "pricerange", "stars", "area", "reference"],
    "attraction": ["price", "type", "address", "postcode", "phone", "area", "reference"],
    "train": ["time", "leave", "price", "arrive", "id", "reference"],
    "restaurant": ["phone", "postcode", "address", "pricerange", "food", "area", "reference"]
}

ALL_REQSLOT = ["car", "address", "postcode", "phone", "internet", "parking", "type", "pricerange", "food",
                      "stars", "area", "reference", "time", "leave", "price", "arrive", "id"]

INFORMABLE_SLOTS = {
    "taxi": ["leave", "destination", "departure", "arrive"],
    "police": [],
    "hospital": ["department"],
    "hotel": ["type", "parking", "pricerange", "internet", "stay", "day", "people", "area", "stars", "name"],
    "attraction": ["area", "type", "name"],
    "train": ["destination", "day", "arrive", "departure", "people", "leave"],
    "restaurant": ["food", "pricerange", "area", "name", "time", "day", "people"]
}

ALL_INFSLOT = ["type", "parking", "pricerange", "internet", "stay", "day", "people", "area", "stars", "name",
               "leave", "destination", "departure", "arrive", "department", "food", "time"]

GOAL_SLOTS = {
    'taxi': [''],
    'police': ['phone', 'postcode', 'address'],
    'hospital': [''],
    'hotel': ['people', 'pricerange', 'stars', 'parking', 'name', 'postcode', 'address', 'area', 'day', 'phone', 'stay', 'type', 'internet'],
    'attraction': [''],
    'train': ['destination', 'duration', 'people', 'leave', 'departure', 'arrive', ],
    'restaurant': [''],
}

EXTRACTIVE_SLOT = ["leave", "arrive", "destination", "departure", "type", "name", "food"]

DA_ABBR_TO_SLOT_NAME = {
    'addr': "address",
    'fee': "price",
    'post': "postcode",
    'ref': 'reference',
    'ticket': 'price',
    'depart': "departure",
    'dest': "destination",
}

DIALOG_ACTS = {
    'restaurant': ['inform', 'request', 'nooffer', 'recommend', 'select', 'offerbook', 'offerbooked', 'nobook'],
    'hotel': ['inform', 'request', 'nooffer', 'recommend', 'select', 'offerbook', 'offerbooked', 'nobook'],
    'attraction': ['inform', 'request', 'nooffer', 'recommend', 'select'],
    'train': ['inform', 'request', 'nooffer', 'offerbook', 'offerbooked', 'select'],
    'taxi': ['inform', 'request'],
    'police': ['inform', 'request'],
    'hospital': ['inform', 'request'],
    'general': ['bye', 'greet', 'reqmore', 'welcome'],
}

USER_ACTS = {
    'general': ['thank'],
}

BOS_USER_TOKEN = "<bos_user>"
EOS_USER_TOKEN = "<eos_user>"

USER_TOKENS = [BOS_USER_TOKEN, EOS_USER_TOKEN]

BOS_BELIEF_TOKEN = "<bos_belief>"
EOS_BELIEF_TOKEN = "<eos_belief>"

BELIEF_TOKENS = [BOS_BELIEF_TOKEN, EOS_BELIEF_TOKEN]

BOS_GOAL_TOEKN = "<bos_goal>"
EOS_GOAL_TOKEN = "<eos_goal>"

GOAL_TOKENS = [BOS_GOAL_TOEKN, EOS_GOAL_TOKEN]

BOS_USER_ACTION_TOKEN = "<bos_user_act>"
EOS_USER_ACTION_TOKEN = "<eos_user_act>"

USER_ACTION_TOEKNS = [BOS_USER_ACTION_TOKEN, EOS_USER_ACTION_TOKEN]

BOS_DB_TOKEN = "<bos_db>"
EOS_DB_TOKEN = "<eos_db>"

DB_TOKENS = [BOS_DB_TOKEN, EOS_DB_TOKEN]

BOS_ACTION_TOKEN = "<bos_act>"
EOS_ACTION_TOKEN = "<eos_act>"

ACTION_TOKENS = [BOS_ACTION_TOKEN, EOS_ACTION_TOKEN]

BOS_RESP_TOKEN = "<bos_resp>"
EOS_RESP_TOKEN = "<eos_resp>"

RESP_TOKENS = [BOS_RESP_TOKEN, EOS_RESP_TOKEN]

DB_NULL_TOKEN = "[db_null]"
DB_0_TOKEN = "[db_0]"
DB_1_TOKEN = "[db_1]"
DB_2_TOKEN = "[db_2]"
DB_3_TOKEN = "[db_3]"

DB_STATE_TOKENS = [DB_NULL_TOKEN, DB_0_TOKEN, DB_1_TOKEN, DB_2_TOKEN, DB_3_TOKEN]

SPECIAL_TOKENS = USER_TOKENS + BELIEF_TOKENS + DB_TOKENS + ACTION_TOKENS + RESP_TOKENS + DB_STATE_TOKENS + GOAL_TOKENS + USER_ACTION_TOEKNS

UBAR_TOKENS = ['<sos_u>', '<eos_u>', '<sos_r>', '<eos_r>', '<sos_b>', '<eos_b>', '<sos_a>', '<eos_a>', '<sos_d>', '<eos_d>']