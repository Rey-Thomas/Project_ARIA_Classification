from torch import nn
import torch
from torch.utils import data
from torchvision.transforms.functional import to_pil_image
import numpy as np
import cv2
import json
import torchmetrics
from model import DataSet_MultiScale
from utils_fcts import polylines, lecture_name_param_image, lecture_json_training, draw_patches_image

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" #due to an error (OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.)

folder_name = 'MultiScale_model_ResNet50_loss_CE_lr_0.8_epochs_150_batch_size_64_use_weight_True_Fine_Tunning_True_multiscaling_input_False/'
model_name = 'model_epochs_149_acc_82'
multi_scale =  True

PATH = f'Model/{folder_name}{model_name}.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device ='cpu'

with open(f'Model/{folder_name}trainning_info.json', 'r') as jsonfile:
        # Reading from json file
        json_dict = json.load(jsonfile)

## plot loss
lecture_json_training(json_dict,folder_name)


#model=torch.load(PATH,map_location=torch.device(device))

state_dict = torch.load(PATH)

from torchvision.models import resnet50
model = resnet50()
model.fc = nn.Linear(in_features=2048, out_features=4)
image_resize = [224,224]

model.load_state_dict(state_dict)
model.to(device)


print('model  eval')
model.eval()
metric = torchmetrics.Accuracy(num_classes=4, average = None).to(device)
f1 = torchmetrics.F1Score(task="multiclass", num_classes=4).to(device)

with open('test.json', 'r') as jsonfile:
        # Reading from json file
        json_dict_test = json.load(jsonfile)


test_dict = {}
correct = 0
liste_image = ['239_Layer00223_Visible_LayeringEnd.jpg', '269_Layer00003_Visible_LayeringEnd.jpg', '437_Layer00041_Visible_LayeringEnd.jpg']
pred_tot = torch.tensor([])
pred_tot = pred_tot.to(device)
label_tot = torch.tensor([])
label_tot = label_tot.to(device)
for name_img in liste_image: # 3 total image in the testset
    path_image = 'D:/ARIA/Dataset/Image/Photo_recadree/' + name_img
    image = cv2.imread(path_image)
    for key in json_dict_test.keys():
        if name_img[:-4] in key:
                json_temp = {key:json_dict_test[key]}
                testset = DataSet_MultiScale(json_temp, image_resize, multi_scale = multi_scale)
                testset_dataloader = data.DataLoader(dataset=testset,
                                            batch_size=1,
                                            shuffle=True)
                image_multiscale, label = next(iter(testset_dataloader))
                image_multiscale = image_multiscale.to(device)
                label = label.to(device)


                output = model(image_multiscale)
                pred = output.data.max(1, keepdim=True)[1]
                pred_tot = torch.cat((pred_tot, pred),dim=0)
                label = label.data.max(1, keepdim=True)[1]
                label_tot = torch.cat((label_tot, label),dim=0)
                test_dict[key] = int(pred)
                #print(f'pred {pred} of type {type(pred)} of size {pred.size()}\n label {label} of type {type(label)} of size {label.size()}')
                correct += pred.eq(label.data.view_as(pred)).cpu().sum()

                name_image, patch_nb_height, patch_nb_width = lecture_name_param_image(key)
            
                image = draw_patches_image(image, output, patch_nb_width, patch_nb_height)


    window_name = 'image'
#     cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)# cv2.WINDOW_NORMAL)
#     cv2.imshow(window_name,image)
#     cv2.waitKey()
    cv2.imwrite(f'Model/{folder_name}/{name_img}', image)

pred_tot = pred_tot.to(torch.int)
label_tot = label_tot.to(torch.int)
weight_accuracy = metric(pred_tot, label_tot)
f1_score = f1(pred_tot, label_tot)
test_dict['accuracy okay'] = 100*float(weight_accuracy[0])
test_dict['accuracy recoater hopping'] = 100*float(weight_accuracy[1])
test_dict['accuracy super elevation'] = 100*float(weight_accuracy[2])
test_dict['accuracy others'] = 100*float(weight_accuracy[3])
test_dict['F1 score'] = float(f1_score)
weighted_acc = (100* (float(weight_accuracy[0])*12053 + float(weight_accuracy[1])*4200 + float(weight_accuracy[2])*1790 + float(weight_accuracy[3])*94)/18142)

test_dict['weight accuracy'] = weighted_acc
test_dict['accuracy test'] = float(100. * correct / len(json_dict_test))
test_dict['name model use'] = model_name
with open(f'Model/{folder_name}test_result.json', 'w') as jsonfile:
        # Reading from json file
        json.dump(test_dict, jsonfile)
print(f'Accuracy Test : {correct}/{len(json_dict_test)} ({100. * correct / len(json_dict_test):.0f}%)')
print(f'Weight Accuracy Test : {weighted_acc:.0f}%)')
print(f'F1 score: {float(f1_score):.3f})')