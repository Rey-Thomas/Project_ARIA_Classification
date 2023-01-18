import argparse
from sklearn.metrics import confusion_matrix, accuracy_score
import json
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Mesuring some score on the test set for a model')
parser.add_argument('--folder', type=str, default='Model/', metavar='F',
                    help="name of the folder in the Model folder where data is located. The trainning_info.json and test_result.json must be in it")
parser.add_argument('--num-classes', type=int, default=4, metavar='C',
                    help="number of classes detected")
 
                   

args = parser.parse_args()


with open(f'D:/MVA/Deep learning/Project/test.json', 'r') as jsonfile:
        # Reading from json file
        label_json_dict = json.load(jsonfile)
with open(f'Model/{args.folder}/test_result.json', 'r') as jsonfile:
        # Reading from json file
        pred_json_dict = json.load(jsonfile)   

pred_list = []
label_list = []
for key in label_json_dict.keys():
    
    if args.num_classes == 4:
        label_name = ["okay","Recoater Hopping", "Super Elevation", "Others"]
        if label_json_dict[key] in [2,3,5]:
            label_list.append(3)
        if label_json_dict[key] == 4:
            label_list.append(2)
        if label_json_dict[key] in [0,1]:
            label_list.append(label_json_dict[key])
    if args.num_classes == 6:
        label_name = ["Okay","Recoater Hopping", "Recoater streaking", "Debris", "Super Elevation", "Incomplete spreading"]
        label_list.append(label_json_dict[key])
    pred_list.append(pred_json_dict[key])

# if 4 in label_list:
#     print('true')

confusion = confusion_matrix(label_list, pred_list)

print(confusion)

fig, ax = plt.subplots()
im = ax.imshow(confusion)
print(type(label_name))
ax.set_xticks(np.arange(len(label_name)))#, labels = label_name)
ax.set_xticklabels(label_name)
ax.set_yticks(np.arange(len(label_name)))#, label_name)
ax.set_yticklabels(label_name)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
for i in range(len(label_name)):
    for j in range(len(label_name)):
        text = ax.text(j, i, confusion[i, j],
                       ha="center", va="center", color="w")
ax.set_title("Confusion Matrix")
fig.tight_layout()
plt.savefig(f'Model/{args.folder}/confusion_matrix.png')
plt.show()