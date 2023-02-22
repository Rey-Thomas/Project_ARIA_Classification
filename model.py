import torch 
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

from torch.utils import data
import os
import json
import cv2
import numpy as np
import time
import torchmetrics
from utils_fcts import lecture_name_param_image

class DataSet_MultiScale(data.Dataset):
    def __init__(self,
                data_json: dict, 
                image_resize: list, #size of the input model
                path_data= 'D:/ARIA/Dataset/Image/Photo_recadree/',
                augment=False,
                transform=None,
                num_classes = 4,
                multi_scale = True
                ):

        self.data_json = data_json 
        self.keys_json = list(self.data_json.keys())
        self.image_resize = image_resize
        self.path_data = path_data
        self.augment=augment
        self.transform =  transform
        self.num_classes = num_classes
        self.multi_scale = multi_scale


    def __len__(self):
        return len(self.data_json)

    def __getitem__(self,
                    index: int):
        # Select the sample
        name_image = self.keys_json[index]
        name_image, patch_nb_height, patch_nb_width = lecture_name_param_image(name_image)

        image = cv2.imread(self.path_data+name_image,cv2.IMREAD_GRAYSCALE)

        imcrop_25_25 = image[25*patch_nb_height:25*(patch_nb_height+1),25*patch_nb_width:25*(patch_nb_width+1)]

        image_pad = cv2.copyMakeBorder(image, 50, 50, 50, 50, cv2.BORDER_REFLECT)

        imcrop_100_100 = image_pad[25*patch_nb_height-37+50:25*(patch_nb_height+1)+38+50,25*patch_nb_width-37+50:25*(patch_nb_width+1)+38+50]
 

        image = torch.tensor(cv2.resize(image , self.image_resize), dtype = torch.float).unsqueeze(0)
        imcrop_25_25 = torch.tensor(cv2.resize(imcrop_25_25, self.image_resize), dtype = torch.float).unsqueeze(0)
        imcrop_100_100 = torch.tensor(cv2.resize(imcrop_100_100, self.image_resize), dtype = torch.float).unsqueeze(0)
        label = torch.zeros([self.num_classes])
        
        if self.num_classes == 6:
            label[int(self.data_json[self.keys_json[index]])] = 1 
        if self.num_classes == 4 :
            if self.data_json[self.keys_json[index]] in [2,3,5]:
                label[3] = 1 
            if self.data_json[self.keys_json[index]] in [0,1]:
                label[int(self.data_json[self.keys_json[index]])] = 1 
            if self.data_json[self.keys_json[index]]==4:
                label[2] = 1 
        if self.multi_scale:
            image_multiscale = torch.cat((imcrop_25_25,imcrop_100_100,image),0)
        if not self.multi_scale:
            image_multiscale = torch.cat((imcrop_25_25,imcrop_25_25,imcrop_25_25),0)
        return image_multiscale, label

if __name__ == '__main__':
    with open('train.json', 'r') as jsonfile:
        # Reading from json file
        json_dict_train = json.load(jsonfile)
    with open('valid.json', 'r') as jsonfile:
        # Reading from json file
        json_dict_valid = json.load(jsonfile)

    # print(f'len valid!!!!!!!!: {len(json_dict_valid)}')
    # print(f'len train!!!!!!!!: {len(json_dict_train)}')


    ###   Parameters to change:   ###

    Fine_Tunning = True
    lr = 0.8#5*10**(-1)
    nb_epochs = 150
    batch_size = 64
    loss_name = 'CE'
    use_weight = True
    name_model = 'ResNet50'
    momentum = 0.9
    multi_scale = False

    if name_model == 'ViT':
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        if not Fine_Tunning:
            for para in model.parameters():
                para.requires_grad = False
        model.heads = nn.Linear(in_features=768, out_features=4)
        image_resize = [384,384]

    if name_model == 'ResNet50':
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        if not Fine_Tunning:
            for para in model.parameters():
                para.requires_grad = False
        model.fc =nn.Linear(in_features=2048, out_features=4)
        image_resize = [224, 224]

    if name_model == 'MobileNet':
        model = mobilenet_v3_large(MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        if not Fine_Tunning:
            for para in model.parameters():
                para.requires_grad = False
        print(model)
        model.classifier = nn.Linear(in_features=960, out_features=4)
        image_resize = [224, 224]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)

    
    loss_dict = {'epochs' : nb_epochs, 'train' : [], 'val' : [], 'accuracy':[], 'weight accuracy':[], 'accuracy okay':[], 'accuracy recoater hopping':[], 'accuracy super elevation':[], 'accuracy others':[]}

    training_dataset=DataSet_MultiScale(json_dict_train, image_resize, multi_scale = multi_scale)
    training_dataloader = data.DataLoader(dataset=training_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)
    valid_dataset=DataSet_MultiScale(json_dict_valid, image_resize, multi_scale = multi_scale)
    valid_dataloader = data.DataLoader(dataset=valid_dataset,
                                            batch_size=batch_size, #len total of valid.json
                                            shuffle=False)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum = momentum) 
    
    weight = torch.tensor([1/12053, 1/4200, 1/1790, 1/94])
    weight = weight.to(device)
    if loss_name == 'BCE':
        criterion = nn.BCEWithLogitsLoss()
    if loss_name == 'CE':
        criterion = nn.CrossEntropyLoss()
        if use_weight:
            criterion = nn.CrossEntropyLoss(weight)

    metric = torchmetrics.Accuracy(num_classes=4, average = None).to(device)
    if multi_scale:
        PATH = f'Model/MultiScale_model_{name_model}_loss_{loss_name}_lr_{lr}_epochs_{nb_epochs}_batch_size_{batch_size}_use_weight_{use_weight}_Fine_Tunning_{Fine_Tunning}/'
        os.mkdir(PATH)
    if not multi_scale:
        PATH = f'Model/MultiScale_model_{name_model}_loss_{loss_name}_lr_{lr}_epochs_{nb_epochs}_batch_size_{batch_size}_use_weight_{use_weight}_Fine_Tunning_{Fine_Tunning}_multiscaling_input_False/'
        os.mkdir(PATH)
    start = time.time()
    for epochs in range(nb_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                mean_loss = 0
                for batch_idx, (image_multiscale, label) in enumerate(training_dataloader):
                    image_multiscale = image_multiscale.to(device)
                    label = label.to(device)
                    optimizer.zero_grad()
                    output = model(image_multiscale)
                    #_ , pred = torch.max(output,1)
                    loss = criterion(output, label)
                    mean_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    if batch_idx % 20 == 0:
                        print(f'Epochs {epochs} [{batch_idx * len(image_multiscale)}/{len(training_dataloader.dataset)} ({100. * batch_idx / len(training_dataloader):.0f}%)]: loss = {loss.item():.6f}')
                print(len(training_dataloader))
                loss_dict[phase].append(mean_loss/len(training_dataloader)) #14 batchs trouver comment trouver le nombre exact tout le temps
                
            if phase == 'val':
                model.eval()
                mean_loss = 0
                correct = 0
                pred_tot = torch.tensor([])
                pred_tot = pred_tot.to(device)
                label_tot = torch.tensor([])
                label_tot = label_tot.to(device)
                for batch_idx, (image_multiscale, label) in enumerate(valid_dataloader):
                    #image_multiscale, label = next(iter(valid_dataloader))
                    image_multiscale = image_multiscale.to(device)
                    label = label.to(device)
                    output = model(image_multiscale)
                    loss = criterion(output, label)
                    mean_loss += loss.item()
                    pred = output.data.max(1, keepdim=True)[1]
                    pred_tot = torch.cat((pred_tot, pred),dim=0)
                    label = label.data.max(1, keepdim=True)[1]
                    label_tot = torch.cat((label_tot, label),dim=0)

                    correct += pred.eq(label.data.view_as(pred)).cpu().sum()

                loss_dict[phase].append(mean_loss/len(valid_dataloader))
                pred_tot = pred_tot.to(torch.int)
                label_tot = label_tot.to(torch.int)
                weight_accuracy = metric(pred_tot, label_tot)
                loss_dict['accuracy okay'].append(100*float(weight_accuracy[0]))
                loss_dict['accuracy recoater hopping'].append(100*float(weight_accuracy[1]))
                loss_dict['accuracy super elevation'].append(100*float(weight_accuracy[2]))
                loss_dict['accuracy others'].append(100*float(weight_accuracy[3]))

                weighted_acc = (100* (float(weight_accuracy[0])*12053 + float(weight_accuracy[1])*4200 + float(weight_accuracy[2])*1790 + float(weight_accuracy[3])*94)/18142)
        print(f'epoch nb {epochs}: train loss = { loss_dict["train"][-1]}, valid loss = { loss_dict["val"][-1] }, Accuracy : {correct}/{len(valid_dataloader.dataset)} ({100. * correct / len(valid_dataloader.dataset):.0f}%), weighted acc: {weighted_acc:.0f}%')
        loss_dict['accuracy'].append(float(100. * correct / len(valid_dataloader.dataset)))
        loss_dict['weight accuracy'].append(weighted_acc)
        
        model_file = PATH + '/model_epochs_' + str(epochs) + '_acc_' + str(int(100. * correct / len(valid_dataloader.dataset))) + '.pth'
        torch.save(model.state_dict(), model_file)
    finish = time.time()


    
    minutes, seconds = divmod(finish-start , 60)
    loss_dict['training time'] = f'{minutes} min'
    loss_dict['name'] = PATH[5:]
    loss_dict['lr'] = lr
    loss_dict['loss used'] = loss_name
    loss_dict['use weight'] = use_weight
    loss_dict['fine tunning'] = Fine_Tunning
    
#saving the json file with a lot of parameters of the training
    if multi_scale:
        with open(f'Model/MultiScale_model_{name_model}_loss_{loss_name}_lr_{lr}_epochs_{nb_epochs}_batch_size_{batch_size}_use_weight_{use_weight}_Fine_Tunning_{Fine_Tunning}/trainning_info.json', 'w') as jsonfile:
            # Reading from json file
            json.dump(loss_dict, jsonfile)
    if not multi_scale:
        with open(f'Model/MultiScale_model_{name_model}_loss_{loss_name}_lr_{lr}_epochs_{nb_epochs}_batch_size_{batch_size}_use_weight_{use_weight}_Fine_Tunning_{Fine_Tunning}_multiscaling_input_False/trainning_info.json', 'w') as jsonfile:
            # Reading from json file
            json.dump(loss_dict, jsonfile)