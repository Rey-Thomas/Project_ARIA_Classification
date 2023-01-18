import json
import random


with open('label.json', 'r') as jsonfile:
    # Reading from json file
    json_dict_label = json.load(jsonfile)

with open('train.json', 'r') as jsonfile:
    # Reading from json file
    json_dict_train = json.load(jsonfile)

with open('valid.json', 'r') as jsonfile:
    # Reading from json file
    json_dict_valid = json.load(jsonfile)

keys = list(json_dict_label.keys())

print(len(keys))
count = 0
random_numbers = random.sample(range(len(keys)), round(0.9*len(keys)))
for i in range(len(keys)):
    if i in random_numbers:
        count += 1 
        json_dict_train[keys[i]] = json_dict_label[keys[i]]
    if i not in random_numbers:
        count += 1 
        json_dict_valid[keys[i]] = json_dict_label[keys[i]]

print(count)

with open('train.json', 'w') as jsonfile:
    # Reading from json file
    json.dump(json_dict_train, jsonfile)

with open('valid.json', 'w') as jsonfile:
    # Reading from json file
    json.dump(json_dict_valid, jsonfile)


