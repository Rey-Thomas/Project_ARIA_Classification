import os
from PIL import Image
import cv2
import json
import random
import numpy as np
from utils_fcts import polylines

nb_ite = 20
nb_build = '437'
testset = True



path_donnee = 'D:/ARIA/Dataset/Image/Photo_recadree/'

# Opening JSON file
with open('test.json', 'r') as jsonfile:
    # Reading from json file
    json_dict = json.load(jsonfile)

# json_dict = json.loads('label.json')

print(f'Number of labeled image: {len(json_dict)}')


print('Labelisation pf 6 classes: \n 0 for "okay" \n 1 for "Recoater hopping" \n 2 for "Recoater streaking" \n 3 for "Debris" \n 4 for "Super elevation" \n 5 for "Incomplete spreading"')
layer = []
#labelisation of an image in particular for the testset
if testset:
    layer_nb = '00041'
    for patch_nb_height in range(38):
        for patch_nb_width in range(38):
            if patch_nb_width==9:
                if int(nb_build)<474: #different image format if the build is old
                    path_image = path_donnee + nb_build + f'_Layer{layer_nb}_Visible_LayeringEnd.jpg'
                if int(nb_build)>=474:
                    path_image = path_donnee + nb_build + f'_Layer{layer_nb}_Visible_LayeringEnd_001.png'
                
                if os.path.exists(path_image):
                    im = cv2.imread(path_image)

                    imcrop = im[ 25*patch_nb_height:25*(patch_nb_height+1),25*patch_nb_width:25*(patch_nb_width+1)]
                    window_name = 'image'


                    imline = polylines(im, patch_nb_height, patch_nb_width) 

                    imline = cv2.resize(imline , [950, 950])
                    imcrop = cv2.resize(imcrop , [100, 100])
                    imcrop = cv2.copyMakeBorder(imcrop, 950//2-50, 950//2-50, 950//2-50, 950//2-50, cv2.BORDER_CONSTANT)
                    #imcrop = cv2.resize(imcrop , [1000, 1000])
                    concat = np.concatenate((imline, imcrop), axis=1)


                    cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
                    cv2.imshow(window_name,concat)
                    label = cv2.waitKey()
                    if label in [ord('0'),ord('1'),ord('2'),ord('3'),ord('4'),ord('5')]:
                        if patch_nb_width>9 and patch_nb_height>9:
                            json_dict[f'{nb_build}_Layer{layer_nb}_Visible_LayeringEnd'+'_'+str(patch_nb_width)+'_'+str(patch_nb_height)  + path_image[-4:]] = int(chr(label))
                        if patch_nb_width<=9 and patch_nb_height>9:
                            json_dict[f'{nb_build}_Layer{layer_nb}_Visible_LayeringEnd'+'_'+'0'+str(patch_nb_width)+'_'+str(patch_nb_height)  + path_image[-4:]] = int(chr(label))
                        if patch_nb_width>9 and patch_nb_height<=9:
                            json_dict[f'{nb_build}_Layer{layer_nb}_Visible_LayeringEnd'+'_'+str(patch_nb_width)+'_'+'0'+str(patch_nb_height)  + path_image[-4:]] = int(chr(label))
                        if patch_nb_width<=9 and patch_nb_height<=9:
                            json_dict[f'{nb_build}_Layer{layer_nb}_Visible_LayeringEnd'+'_'+'0'+str(patch_nb_width)+'_'+'0'+str(patch_nb_height)  + path_image[-4:]] = int(chr(label))


        
if not testset:
    for ite in range(nb_ite):
        layer_nb =random.randint(0, 1841)
        if layer_nb<10:
            layer.append(f'0000{layer_nb}')
        if 9<layer_nb<100:
            layer.append(f'000{layer_nb}')
        if 99<layer_nb<1000:
            layer.append(f'00{layer_nb}')

    for ite in range(nb_ite):
        patch_nb_width = random.randint(0, 37)
        patch_nb_height = random.randint(0, 37)

        layer_nb_int =random.randint(0, 450)
        if layer_nb_int<10:
            layer_nb = f'0000{layer_nb_int}'
        if 9<layer_nb_int<100:
            layer_nb = f'000{layer_nb_int}'
        if 99<layer_nb_int<1000:
            layer_nb = f'00{layer_nb_int}'



        if int(nb_build)<474:
            path_image = path_donnee + nb_build + f'_Layer{layer_nb}_Visible_LayeringEnd.jpg'
        if int(nb_build)>=474:
            path_image = path_donnee + nb_build + f'_Layer{layer_nb}_Visible_LayeringEnd_001.png'
        
        if os.path.exists(path_image):
            im = cv2.imread(path_image)

            imcrop = im[ 25*patch_nb_height:25*(patch_nb_height+1),25*patch_nb_width:25*(patch_nb_width+1)]
            window_name = 'image'

            # pts = np.array([[25*patch_nb_width, 25*patch_nb_height],[25*patch_nb_width, 25*(patch_nb_height+1)], [25*(patch_nb_width+1), 25*(patch_nb_height+1)],[25*(patch_nb_width+1), 25*patch_nb_height]])
            # pts = pts.reshape((-1, 1, 2))
            # imline = cv2.polylines(im, pts,isClosed=True,color = (0, 0, 255),thickness = 5)

            imline = polylines(im, patch_nb_height, patch_nb_width) #remplacer par la fonction 

            imline = cv2.resize(imline , [950, 950])
            imcrop = cv2.resize(imcrop , [100, 100])
            imcrop = cv2.copyMakeBorder(imcrop, 950//2-50, 950//2-50, 950//2-50, 950//2-50, cv2.BORDER_CONSTANT)
            #imcrop = cv2.resize(imcrop , [1000, 1000])
            concat = np.concatenate((imline, imcrop), axis=1)


            cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
            cv2.imshow(window_name,concat)
            label = cv2.waitKey()
            if label in [ord('0'),ord('1'),ord('2'),ord('3'),ord('4'),ord('5')]:
                if patch_nb_width>9 and patch_nb_height>9:
                    json_dict[f'{nb_build}_Layer{layer_nb}_Visible_LayeringEnd'+'_'+str(patch_nb_width)+'_'+str(patch_nb_height)  + path_image[-4:]] = int(chr(label)) #j'ai change la fin pour qu' on ai pas de pb avec les formats png et jpg
                if patch_nb_width<9 and patch_nb_height>9:
                    json_dict[f'{nb_build}_Layer{layer_nb}_Visible_LayeringEnd'+'_'+'0'+str(patch_nb_width)+'_'+str(patch_nb_height)  + path_image[-4:]] = int(chr(label))
                if patch_nb_width>9 and patch_nb_height<9:
                    json_dict[f'{nb_build}_Layer{layer_nb}_Visible_LayeringEnd'+'_'+str(patch_nb_width)+'_'+'0'+str(patch_nb_height)  + path_image[-4:]] = int(chr(label))
                if patch_nb_width<9 and patch_nb_height<9:
                    json_dict[f'{nb_build}_Layer{layer_nb}_Visible_LayeringEnd'+'_'+'0'+str(patch_nb_width)+'_'+'0'+str(patch_nb_height)  + path_image[-4:]] = int(chr(label))



with open('test.json', 'w') as jsonfile:
    # Reading from json file
    json.dump(json_dict, jsonfile)
    #   json.dumps(json_dict)
