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


#0:4677, 1:2515, 2:49, 3:10, 4:594, 5:0, tot= 7846

#13377, 4701, 90, 10, 1979, 1