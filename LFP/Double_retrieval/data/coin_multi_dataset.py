from torch.utils.data import Dataset
import torch
import numpy as np
import random
import cv2
import json
import os
from PIL import Image
from data.coin30_custom import step_label

class Coin_Retrieve_Dataset_with_Taskname_Multi(Dataset):
    def __init__(self,json_file,transform):
        with open(json_file,'r',encoding='utf-8') as f:
            self.data_list = json.load(f)
        self.transform = transform
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        try:
            start_image = Image.open(data['start_image_path_m']).convert('RGB')    
            start_image = self.transform(start_image) 
            end_image = Image.open(data['end_image_path_m']).convert('RGB')    
            end_image = self.transform(end_image)
            start_caption = data['start_caption']
            end_caption = data['end_caption']
            Task_description = data['task_discription']
            interval = data['interval']

            middle_caption_1 = data['interval_step_1']
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

        return description, start_image, end_image, step_label[start_caption], step_label[end_caption],step_label[middle_caption_1]

if __name__ == '__main__':
    pass