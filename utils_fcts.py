import cv2
import numpy as np
import torch

import matplotlib.pyplot as plt



def polylines(image, patch_nb_height, patch_nb_width):
    pts = np.array([[25*patch_nb_width, 25*patch_nb_height],[25*patch_nb_width, 25*(patch_nb_height+1)], [25*(patch_nb_width+1), 25*(patch_nb_height+1)],[25*(patch_nb_width+1), 25*patch_nb_height]])
    pts = pts.reshape((-1, 1, 2))
    imline = cv2.polylines(image, pts,isClosed=True,color = (0, 0, 255),thickness = 5)
    return imline

def lecture_name_param_image(key):
    patch_nb_height = int(key[-6:-4])
    patch_nb_width = int(key[-9:-7])
    name_image = key[:-10]
    if int(name_image[:3]) >= 474:
        name_image = key[:-10] + '_001' + key[-4:]
    if int(name_image[:3]) < 474:
        name_image = key[:-10]+key[-4:]
    return name_image, patch_nb_height, patch_nb_width

def lecture_json_training(json_dict,model_name):
    if 'accuracy' in json_dict.keys():
        fig, ax1 = plt.subplots()
        lns1 = ax1.plot(np.linspace(0,json_dict['epochs'],json_dict['epochs']),json_dict['train'], color='blue', label='train')
        lns2 =ax1.plot(np.linspace(0,json_dict['epochs'],json_dict['epochs']),json_dict['val'], color='green', label='valid')
        ax2  = ax1.twinx()
        lns3 = ax2.plot(np.linspace(0,json_dict['epochs'],json_dict['epochs']),json_dict['accuracy'], color='red', label='accuracy')
        lns = lns1+lns2+lns3
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc=0)
        ax1.set_ylabel('loss')
        ax1.set_xlabel('epoch')
        ax2.set_ylabel('accuracy (%)')
        plt.title(f'Training Time: {json_dict["training time"]}  Learning rate: {json_dict["lr"]}')
        plt.savefig(f'Model/{model_name}/loss.png')

        plt.show()
    if 'accuracy okay' in json_dict.keys():
        plt.figure()
        plt.plot(np.linspace(0,json_dict['epochs'],json_dict['epochs']),json_dict['accuracy okay'], color='green')
        plt.plot(np.linspace(0,json_dict['epochs'],json_dict['epochs']),json_dict['accuracy recoater hopping'], color='red')
        plt.plot(np.linspace(0,json_dict['epochs'],json_dict['epochs']),json_dict['accuracy super elevation'], color='blue')
        plt.plot(np.linspace(0,json_dict['epochs'],json_dict['epochs']),json_dict['accuracy others'], color='pink')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend(['okay','recoater hopping','super elevation','others'])
        plt.title(f'Training Time: {json_dict["training time"]}  Learning rate: {json_dict["lr"]}')
        plt.savefig(f'Model/{model_name}/accuracy_label.png')
        plt.show()
    
    else:
        fig, ax1 = plt.figure()
        plt.plot(np.linspace(0,json_dict['epochs'],json_dict['epochs']),json_dict['train'], color='blue')
        plt.plot(np.linspace(0,json_dict['epochs'],json_dict['epochs']),json_dict['val'], color='green')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend(['train','val'])
        plt.title(f'Training Time: {json_dict["training time"]}  Learning rate: {json_dict["lr"]}')
        plt.savefig(f'Model/{model_name}/loss.png')
        plt.show()
    

def draw_patches_image(image, output, patch_nb_height, patch_nb_width):
    #dict_color = {'0':(0,255,0), '1':(0,0,255), '2':(255, 128, 0), '3':(125,125,0), '4':(255,0,0), '5':(255, 153, 255)} # green, red, cyan, orange, blue, pink #6 classes
    dict_color = {'0':(0,255,0), '1':(0,0,255), '2':(255,0,0), '3':(255, 153, 255)} # green, red, blue, pink #4 classes
    max , prediction = torch.max(output,1)
    color = dict_color[str(int(prediction[0]))]
    image = cv2.rectangle(image, (25*patch_nb_height, 25*patch_nb_width), (25*(patch_nb_height+1), 25*(patch_nb_width+1)), color, thickness = 1)
    ###to use for BCE
    #image = cv2.putText(image, str(int(max*100)), org = (25*patch_nb_height, 25*(patch_nb_width+1)), fontFace = cv2.FONT_HERSHEY_PLAIN, fontScale = 1, color=color)
    return image
    


