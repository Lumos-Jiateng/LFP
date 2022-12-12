from torch.utils.data import Dataset
import torch
import numpy as np
import random
import cv2
import json
import os
from data.coin180_custom import step_list_180, label_step_180, step_label_180
from PIL import Image
from data.crosstask_105 import step_label_105,step_list_105,label_step_105

# Train: Train the double retrieval with single / multi-imgae / limited range of data.
class Coin_Infer_Train(Dataset):
    def __init__(self,json_file,transform,head_number):
        with open(json_file,'r',encoding='utf-8') as f:
            self.data_list = json.load(f)
        self.transform = transform
        self.head_number = head_number

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
            return self.__getitem__(0)

        if interval == 3:
            description = 'The Task is ' + Task_description + ' and there are three steps in between .'
        elif interval == 2:
            description = 'The Task is ' + Task_description + ' and there are two steps in between .'
        elif interval == 1:
            description = 'The Task is ' + Task_description + ' and there is only one step in between .'
        elif interval == 0:
            description = 'The Task is ' + Task_description + ' and there is no step in between .'
        
        if self.head_number == 2:
            return description, start_image, end_image, step_label_180[start_caption], step_label_180[end_caption]
        
        elif self.head_number == 3:
            try:
                step_1_image = Image.open(data['interval_1_image_path']).convert('RGB')    
                step_1_image = self.transform(start_image) 
                step_1_caption = data['interval_step_1']
            except BaseException as e:
                print('three head error')
                return self.__getitem__(0)
            return description, start_image,step_1_image,end_image, step_label_180[start_caption],step_label_180[step_1_caption],step_label_180[end_caption]
        
        elif self.head_number == 4:
            try:
                step_1_image = Image.open(data['interval_1_image_path']).convert('RGB')    
                step_1_image = self.transform(start_image) 
                step_1_caption = data['interval_step_1']
                step_2_image = Image.open(data['interval_2_image_path']).convert('RGB')    
                step_2_image = self.transform(start_image) 
                step_2_caption = data['interval_step_2']
            except BaseException as e:
                print('four head error')
                return self.__getitem__(0)
            return description, start_image,step_1_image,step_2_image,end_image, step_label_180[start_caption],step_label_180[step_1_caption],step_label_180[step_2_caption], step_label_180[end_caption]

class Coin_Infer_Test(Dataset):
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
            return self.__getitem__(0)
 
        # Never return here
        if interval == 3:
            step_1_caption = data['interval_step_1']
            step_2_caption = data['interval_step_2']
            step_3_caption = data['interval_step_3']
            description = 'The Task is ' + Task_description + ' and there are three steps in between .'
            return description, start_image, end_image, step_label_180[start_caption],step_1_caption,step_2_caption,step_3_caption,step_label_180[end_caption]
        elif interval == 2:
            step_1_caption = data['interval_step_1']
            step_2_caption = data['interval_step_2']
            description = 'The Task is ' + Task_description + ' and there are two steps in between .'
            return description, start_image, end_image, step_label_180[start_caption],step_1_caption,step_2_caption,step_label_180[end_caption]
        elif interval == 1:
            step_1_caption = data['interval_step_1']
            description = 'The Task is ' + Task_description + ' and there is only one step in between .'
            return description, start_image, end_image, step_label_180[start_caption],step_1_caption,step_label_180[end_caption]
        # Never return here
        elif interval == 0:
            description = 'The Task is ' + Task_description + ' and there is no step in between .'
            return description, start_image, end_image, step_label_180[start_caption], step_label_180[end_caption]
        

class Crosstask_Infer_Train(Dataset):
    def __init__(self,json_file,transform,head_number):
        with open(json_file,'r',encoding='utf-8') as f:
            self.data_list = json.load(f)
        self.transform = transform
        self.head_number = head_number

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
            if(Task_description) == 'changing_tire':
                Task_description = 'Change a Tire'
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
        
        if self.head_number == 2:
            return description, start_image, end_image, step_label_105[start_caption], step_label_105[end_caption]
        
        elif self.head_number == 3:
            try:
                step_1_image = Image.open(data['interval_1_image_path']).convert('RGB')    
                step_1_image = self.transform(start_image) 
                step_1_caption = data['interval_step_1']
            except BaseException as e:
                print('three head error')
                return self.__getitem__(0)
            return description, start_image,step_1_image,end_image, step_label_105[start_caption],step_label_105[step_1_caption],step_label_105[end_caption]
        
        elif self.head_number == 4:
            try:
                step_1_image = Image.open(data['interval_1_image_path']).convert('RGB')    
                step_1_image = self.transform(start_image) 
                step_1_caption = data['interval_step_1']
                step_2_image = Image.open(data['interval_2_image_path']).convert('RGB')    
                step_2_image = self.transform(start_image) 
                step_2_caption = data['interval_step_2']
            except BaseException as e:
                print('four head error')
                return self.__getitem__(0)
            return description, start_image,step_1_image,step_2_image,end_image, step_label_105[start_caption],step_label_105[step_1_caption],step_label_105[step_2_caption], step_label_105[end_caption]

class Crosstask_Infer_Test(Dataset):
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
            if(Task_description) == 'changing_tire':
                Task_description = 'Change a Tire'
            interval = data['interval']
        except BaseException as e:
            print('error')
            return self.__getitem__(0)
 
        # Never return here
        if interval == 3:
            description = 'The Task is ' + Task_description + ' and there are three steps in between .'
            return description, start_image, end_image, step_label_105[start_caption], step_label_105[end_caption]
        elif interval == 2:
            step_1_caption = data['interval_step_1']
            step_2_caption = data['interval_step_2']
            description = 'The Task is ' + Task_description + ' and there are two steps in between .'
            return description, start_image, end_image, step_label_105[start_caption],step_1_caption,step_2_caption,step_label_105[end_caption]
        elif interval == 1:
            step_1_caption = data['interval_step_1']
            description = 'The Task is ' + Task_description + ' and there is only one step in between .'
            return description, start_image, end_image, step_label_105[start_caption],step_1_caption,step_label_105[end_caption]
        # Never return here
        elif interval == 0:
            description = 'The Task is ' + Task_description + ' and there is no step in between .'
            return description, start_image, end_image, step_label_105[start_caption], step_label_105[end_caption]        






class Image_Bart_Dataset(Dataset):
    def __init__(self,json_file,transform,head_number):
        with open(json_file,'r',encoding='utf-8') as f:
            self.data_list = json.load(f)
        assert(head_number == 2)
        self.transform = transform
        self.head_number = head_number

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
            return self.__getitem__(0)
        
        try:
            if interval == 1:
                description_0 = 'For Task ' + Task_description + ', Given the first step and the last step, predict the intermediate one step. '
                step_1_caption = data['interval_step_1']
            elif interval == 2:
                description_0 = 'For Task ' + Task_description + ', Given the first step and the last step, predict the intermediate two steps. '
                step_1_caption = data['interval_step_1']
                step_2_caption = data['interval_step_2']
            elif interval == 3:
                description_0 = 'For Task ' + Task_description + ', Given the first step and the last step, predict the intermediate two steps. '
                step_1_caption = data['interval_step_1']
                step_2_caption = data['interval_step_2']
                step_3_caption = data['interval_step_3']
        except BaseException as e:
            print(data)
            print('error in intermedate steps')
            return self.__getitem__(0)
        
        if interval == 1:
            return description_0,start_image,end_image,start_caption,step_1_caption,end_caption
        
        if interval == 2:
            step_caption = step_1_caption+ '.' + step_2_caption + '.'
            return description_0,start_image,end_image,start_caption,step_caption,end_caption

        if interval == 3:
            step_caption = step_1_caption+ '.' + step_2_caption + '.' + step_3_caption + '.'
            return description_0,start_image,end_image,start_caption,step_caption,end_caption
        
# Test: Test for the whole prediction: image*2 -> K steps
class Image_Bart_Dataset_Predict_all(Dataset):
    def __init__(self,json_file,transform,head_number):
        with open(json_file,'r',encoding='utf-8') as f:
            self.data_list = json.load(f)
        assert(head_number == 2)
        self.transform = transform
        self.head_number = head_number

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
            return self.__getitem__(0)
        
        try:
            if interval == 1:
                description_0 = 'For Task ' + Task_description + ', Given the first step and the last step, predict the intermediate one step. '
                step_1_caption = data['interval_step_1']
            elif interval == 2:
                description_0 = 'For Task ' + Task_description + ', Given the first step and the last step, predict the intermediate two steps. '
                step_1_caption = data['interval_step_1']
                step_2_caption = data['interval_step_2']
            elif interval == 3:
                description_0 = 'For Task ' + Task_description + ', Given the first step and the last step, predict the intermediate two steps. '
                step_1_caption = data['interval_step_1']
                step_2_caption = data['interval_step_2']
                step_3_caption = data['interval_step_3']
        except BaseException as e:
            print(data)
            print('error in intermedate steps')
            return self.__getitem__(0)
        
        if interval == 1:
            # add a line to include start / end into caption supervision
            step_1_caption = start_caption + '.' + end_caption + '.' + step_1_caption + '.' 
            return description_0,start_image,end_image,start_caption,step_1_caption,end_caption
        
        if interval == 2:
            step_caption = step_1_caption+ '.' + step_2_caption + '.'
            step_caption = start_caption + '.' + end_caption + '.' + step_caption
            return description_0,start_image,end_image,start_caption,step_caption,end_caption

        if interval == 3:
            step_caption = start_caption + '.' + end_caption + '.' + step_caption
            step_caption = step_1_caption+ '.' + step_2_caption + '.' + step_3_caption + '.'
            return description_0,start_image,end_image,start_caption,step_caption,end_caption


class Image_Bart_Dataset_all(Dataset):
    def __init__(self,json_file,transform,head_number):
        with open(json_file,'r',encoding='utf-8') as f:
            self.data_list = json.load(f)
        assert(head_number == 2)
        self.transform = transform
        self.head_number = head_number
        self.first_right = 0

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
            print(f'error happens for index {index},self.first_right = {self.first_right}')
            if index != self.first_right:
                return self.__getitem__(self.first_right)
            else:
                self.first_right = self.first_right + 1
                return self.__getitem__(self.first_right)
        
        try:
            if interval == 1:
                description_0 = 'For Task ' + Task_description + ', Given the first step and the last step, predict the intermediate one step. '
                step_caption = start_caption + '.' + data['interval_step_1'] + '.' + end_caption + '.'
            elif interval == 2:
                description_0 = 'For Task ' + Task_description + ', Given the first step and the last step, predict the intermediate two steps. '
                #step_caption = data['interval_step_1']
                #step_caption = data['interval_step_2']
                step_caption = start_caption + '.' + data['interval_step_1'] + '.' + data['interval_step_2'] + '.' + end_caption + '.'
            elif interval == 3:
                description_0 = 'For Task ' + Task_description + ', Given the first step and the last step, predict the intermediate two steps. '
                #step_caption = data['interval_step_1']
                #step_caption = data['interval_step_2']
                step_caption = start_caption + '.' + data['interval_step_1'] + '.' + data['interval_step_2'] + '.' + data['interval_step_3'] + '.' + end_caption + '.'
        except BaseException as e:
            print(data)
            print('error in intermedate steps')
            return self.__getitem__(0)
        
        if interval == 1:
            return description_0,start_image,end_image,step_caption
        
        if interval == 2:
            return description_0,start_image,end_image,step_caption

        if interval == 3:
            return description_0,start_image,end_image,step_caption

class Image_Bart_Dataset_Predict_all_123(Dataset):
    def __init__(self,json_file,transform,head_number):
        with open(json_file,'r',encoding='utf-8') as f:
            self.data_list = json.load(f)
        assert(head_number == 2)
        self.transform = transform
        self.head_number = head_number

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
            return self.__getitem__(0)
        
        try:
            if interval == 1:
                description_0 = 'For Task ' + Task_description + ', Given the first step and the last step, predict the intermediate one step. '
                step_1_caption = data['interval_step_1']
            elif interval == 2:
                description_0 = 'For Task ' + Task_description + ', Given the first step and the last step, predict the intermediate two steps. '
                step_1_caption = data['interval_step_1']
                step_2_caption = data['interval_step_2']
            elif interval == 3:
                description_0 = 'For Task ' + Task_description + ', Given the first step and the last step, predict the intermediate two steps. '
                step_1_caption = data['interval_step_1']
                step_2_caption = data['interval_step_2']
                step_3_caption = data['interval_step_3']
        except BaseException as e:
            print(data)
            print('error in intermedate steps')
            return self.__getitem__(0)
        
        if interval == 1:
            # add a line to include start / end into caption supervision
            step_1_caption = start_caption + '.' + step_1_caption + '.' + end_caption + '.'
            return description_0,start_image,end_image,start_caption,step_1_caption,end_caption
        
        if interval == 2:
            step_caption = step_1_caption+ '.' + step_2_caption + '.'
            step_caption = start_caption + '.' + step_caption + end_caption + '.'
            return description_0,start_image,end_image,start_caption,step_caption,end_caption

        if interval == 3:
            step_caption = step_1_caption+ '.' + step_2_caption + '.' + step_3_caption + '.'
            tep_caption = start_caption + '.' + step_caption + end_caption + '.'
            return description_0,start_image,end_image,start_caption,step_caption,end_caption