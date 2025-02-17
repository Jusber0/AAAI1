import pickle

with open(r"data/JAAD_data/beh_seq_test.pkl", 'rb') as f:
    data = pickle.load(f)

bbox_data = data['bbox']
intent_data = data['intent']

for bbox in data['bbox']:
    for frame in bbox:
        print(frame)

for intent in data['intent']:
    for frame in intent:
        print(frame)
