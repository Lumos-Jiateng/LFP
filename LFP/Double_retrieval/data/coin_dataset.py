from torch.utils.data import Dataset
import torch
import numpy as np
import random
import cv2
import json
import os
from PIL import Image
from data.coin180_custom import step_label_180
from data.crosstask_105 import step_label_105

class Coin_Retrieve_Dataset_with_Taskname_Double(Dataset):
    def __init__(self,json_file,transform):
        with open(json_file,'r',encoding='utf-8') as f:
            self.data_list = json.load(f)
        self.transform = transform
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        try:
            start_image = Image.open(data['start_image_path']).convert('RGB')    
            start_image = self.transform(start_image) 
            end_image = Image.open(data['end_image_path']).convert('RGB')    
            end_image = self.transform(end_image)
            start_caption = data['start_caption']
            end_caption = data['end_caption']
            Task_description = data['task_discription']
            interval = data['interval']
        except BaseException as e:
            print('error')
            print(data)
            return self.__getitem__(0)

        if interval == 3:
            description = 'The Task is ' + Task_description + ' and there are three steps in between .'
        elif interval == 2:
            description = 'The Task is ' + Task_description + ' and there are two steps in between .'
        elif interval == 1:
            description = 'The Task is ' + Task_description + ' and there is only one step in between .'
        elif interval == 0:
            description = 'The Task is ' + Task_description + ' and there is no step in between .'
        elif interval == 4:
            description = 'The Task is ' + Task_description + ' and there are four steps in between .'
        
        return description, start_image, end_image, step_label_180[start_caption], step_label_180[end_caption]
    

class Crosstask_Retrieve_Dataset_with_Taskname_Double(Dataset):
    def __init__(self,json_file,transform):
        with open(json_file,'r',encoding='utf-8') as f:
            self.data_list = json.load(f)
        self.transform = transform
        self.error_number = 0
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        try:
            start_image = Image.open(data['start_image_path']).convert('RGB')    
            start_image = self.transform(start_image) 
            end_image = Image.open(data['end_image_path']).convert('RGB')    
            end_image = self.transform(end_image)
            start_caption = data['start_caption']
            end_caption = data['end_caption']
            Task_description = data['task_discription']
            interval = data['interval']
            #print(data)
        except BaseException as e:
            print('error crosstask')
            self.error_number = self.error_number+1
            print(self.error_number)
            return self.__getitem__(index+1)

        if interval == 3:
            description = 'The Task is ' + Task_description + ' and there are three steps in between .'
        elif interval == 2:
            description = 'The Task is ' + Task_description + ' and there are two steps in between .'
        elif interval == 1:
            description = 'The Task is ' + Task_description + ' and there is only one step in between .'
        elif interval == 0:
            description = 'The Task is ' + Task_description + ' and there is no step in between .'

        return description, start_image, end_image, step_label_105[start_caption], step_label_105[end_caption]


class Coin_Retrieve_Dataset_without_Taskname_Double(Dataset):
    def __init__(self,json_file,transform):
        with open(json_file,'r',encoding='utf-8') as f:
            self.data_list = json.load(f)
        self.transform = transform
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        try:
            start_image = Image.open(data['start_image_path']).convert('RGB')    
            start_image = self.transform(start_image) 
            end_image = Image.open(data['end_image_path']).convert('RGB')    
            end_image = self.transform(end_image)
            start_caption = data['start_caption']
            end_caption = data['end_caption']
            Task_description = data['task_discription']
            interval = data['interval']
        except BaseException as e:
            print('error')
            print(data)
            return self.__getitem__(0)

        if interval == 3:
            description = 'The Task is ' + 'unknown' + ' and there are three steps in between .'
        elif interval == 2:
            description = 'The Task is ' + 'unknown' + ' and there are two steps in between .'
        elif interval == 1:
            description = 'The Task is ' + 'unknown' + ' and there is only one step in between .'
        elif interval == 0:
            description = 'The Task is ' + 'unknown' + ' and there is no step in between .'

        return description, start_image, end_image, step_label_180[start_caption], step_label_180[end_caption]

class Coin_Retrieve_Dataset_with_Taskname_single_infer(Dataset):
    def __init__(self,json_file,transform):
        with open(json_file,'r',encoding='utf-8') as f:
            self.data_list = json.load(f)
        self.transform = transform
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        try:
            start_image = Image.open(data['start_image_path_s']).convert('RGB') 
            start_image = np.array(start_image) 
            #print(start_image.shape)
            start_image = expand(start_image)
            #print(start_image.shape)
            start_image = Image.fromarray(start_image) 
            start_image = self.transform(start_image) 

            end_image = Image.open(data['end_image_path_s']).convert('RGB')    
            end_image = np.array(end_image) 
            #print(end_image.shape)
            end_image = expand(end_image)
            #print(end_image.shape)
            end_image = Image.fromarray(end_image)
            end_image = self.transform(end_image)

            start_caption = data['start_caption']
            end_caption = data['end_caption']
            Task_description = data['task_discription']
            interval = data['interval']
        except BaseException as e:
            print('error')
            return self.__getitem__(0)

        if interval == 3:
            description = 'The Task is ' + Task_description + ' and there are three steps in between .'
        elif interval == 2:
            description = 'The Task is ' + Task_description + ' and there are two steps in between .'
        elif interval == 1:
            description = 'The Task is ' + Task_description + ' and there is only one step in between .'
        elif interval == 0:
            description = 'The Task is ' + Task_description + ' and there is no step in between .'

        return description, start_image, end_image, step_label_180[start_caption], step_label_180[end_caption]

def image_combine(self,img_frms):
    assert(img_frms.shape[0] == 9)
    img_frm_line1=torch.cat((img_frms[0,:,:,:],img_frms[1,:,:,:],img_frms[2,:,:,:]),1)
    img_frm_line2=torch.cat((img_frms[3,:,:,:],img_frms[4,:,:,:],img_frms[5,:,:,:]),1)
    img_frm_line3=torch.cat((img_frms[6,:,:,:],img_frms[7,:,:,:],img_frms[8,:,:,:]),1)
    return torch.cat((img_frm_line1,img_frm_line2,img_frm_line3),2)        

def expand(img):
    line = np.concatenate([img,img,img],axis=0)
    new_image = np.concatenate([line,line,line],axis=1)
    return new_image

if __name__ == '__main__':
    pass