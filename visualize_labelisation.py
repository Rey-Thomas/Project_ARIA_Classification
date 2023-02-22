import torch
import cv2
import json
from torch.utils import data
from utils_fcts import polylines, lecture_name_param_image, lecture_json_training, draw_patches_image
from model import DataSet_MultiScale


with open('test.json', 'r') as jsonfile:
        # Reading from json file
        json_dict_test = json.load(jsonfile)

test_dict = {}
correct = 0
liste_image = ['239_Layer00223_Visible_LayeringEnd.jpg', '269_Layer00003_Visible_LayeringEnd.jpg', '437_Layer00041_Visible_LayeringEnd.jpg']
image_resize = [224,224]

for name_img in liste_image: 
    path_image = 'D:/ARIA/Dataset/Image/Photo_recadree/' + name_img
    image = cv2.imread(path_image)
    for key in json_dict_test.keys():
        if name_img[:-4] in key:

                json_temp = {key:json_dict_test[key]}
                testset = DataSet_MultiScale(json_temp, image_resize)
                testset_dataloader = data.DataLoader(dataset=testset,
                                            batch_size=1,
                                            shuffle=True)
                image_multiscale, label = next(iter(testset_dataloader))

                name_image, patch_nb_height, patch_nb_width = lecture_name_param_image(key)
            
                image = draw_patches_image(image, label, patch_nb_width,  patch_nb_height)

    window_name = 'image'
#     cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)# cv2.WINDOW_NORMAL)
#     cv2.imshow(window_name,image)
#     cv2.waitKey()
    cv2.imwrite(f'Model/{name_img[:-4]}_label.jpg', image)