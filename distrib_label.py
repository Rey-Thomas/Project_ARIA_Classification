import json
import matplotlib.pyplot as plt


# Opening JSON file
with open('label.json', 'r') as jsonfile:
    # Reading from json file
    json_dict = json.load(jsonfile) 


print(len(json_dict))
plt.figure()
print(plt.hist(json_dict.values(),align = 'left'))
plt.hist(json_dict.values(),align = 'left')
plt.xlabel('label')
plt.ylabel('number of patch')
plt.show()
