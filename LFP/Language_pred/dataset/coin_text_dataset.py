import json
import os
import random

from torch.utils.data import Dataset

def Number_to_String(number):
    if number == 0:
        return 'zero'
    if number == 1:
        return 'one'
    if number == 2:
        return 'two'
    if number == 3:
        return 'three'
    if number == 4:
        return 'four'
    if number == 5:
        return 'five'
    if number == 6:
        return 'six'
    if number == 7:
        return 'seven'
    if number == 8:
        return 'eight'
    if number == 9:
        return 'nine'
    if number == 10:
        return 'ten'
    else:
        return 'unknown'

def Add_Prompt_0(number,start_string,goal_string):
    return 'Taking ' + Number_to_String(number) + ' steps, ' + 'from ' + start_string + ' to ' + goal_string + ', we need to '

def Add_Prompt_1(number,start_string,goal_string):
    return 'You start from '+ start_string +'. Your goal is ' + goal_string + '. List ' + Number_to_String(number) +' steps to do this.'

def Add_Prompt_for_Image_Bart(Task_description,number,start_string,goal_string):
    return 'For Task ' + Task_description + ', Given the first step and the last step, predict the intermediate ' + Number_to_String(number) + ' step.' + start_string + '.' + goal_string + '.'

def Add_Prompt_Task_description(Task_description,start_caption,end_caption,predict_number):
    return 'For Task ' + Task_description + ', The first image is about ' + start_caption + ', The second image is about ' + end_caption+ ', predict the next ' + Number_to_String(predict_number) + ' steps.'

class coin_text_dataset(Dataset):
    def __init__(self, filename , mode = 0):       
        self.filename=filename 
        with open(self.filename,'r',encoding='utf-8') as file:
             self.action_list=json.load(file)
        self.prompt_mode = mode

    def __len__(self):
        return len(self.action_list)
    
    def __getitem__(self, index):    
        action=self.action_list[index]
        total_step=action['Action_Len']
        predict_step_number=total_step-2
        Start_Message=action['Action_Label'][0]
        Goal_Message=action['Action_Label'][total_step-1]
        Task_decription = action['Target_Label']
        #Predicted='[sos]'
        Predict=""
        for i in range(1,total_step-1):
            Predict=Predict+action['Action_Label'][i]+'.'
        if self.prompt_mode == 0:
            Prompt=Add_Prompt_0(predict_step_number,Start_Message,Goal_Message)
        elif self.prompt_mode == 1:
            Prompt=Add_Prompt_1(predict_step_number,Start_Message,Goal_Message)
        elif self.prompt_mode == 2:
            Prompt=Add_Prompt_for_Image_Bart(Task_decription,predict_step_number,Start_Message,Goal_Message)
        return Prompt,Predict

class coin_text_dataset_all(Dataset):
    def __init__(self, filename , mode = 0):       
        self.filename=filename 
        with open(self.filename,'r',encoding='utf-8') as file:
             self.action_list=json.load(file)
        self.prompt_mode = mode

    def __len__(self):
        return len(self.action_list)
    
    def __getitem__(self, index):    
        action=self.action_list[index]
        total_step=action['Action_Len']
        predict_step_number=total_step-2
        Start_Message=action['Action_Label'][0]
        Goal_Message=action['Action_Label'][total_step-1]
        Task_decription = action['Target_Label']
        #Predicted='[sos]'
        Predict=""
        Predict = action['Action_Label'][0] + '.' + action['Action_Label'][total_step-1] + '.'
        for i in range(1,total_step-1):
            Predict=Predict+action['Action_Label'][i]+'.'
        if self.prompt_mode == 0:
            Prompt=Add_Prompt_0(predict_step_number,Start_Message,Goal_Message)
        elif self.prompt_mode == 1:
            Prompt=Add_Prompt_1(predict_step_number,Start_Message,Goal_Message)
        elif self.prompt_mode == 2:
            Prompt=Add_Prompt_for_Image_Bart(Task_decription,predict_step_number,Start_Message,Goal_Message)
        return Prompt,Predict

class coin_text_dataset_all_213(Dataset):
    def __init__(self, filename , mode = 0):       
        self.filename=filename 
        with open(self.filename,'r',encoding='utf-8') as file:
             self.action_list=json.load(file)
        self.prompt_mode = mode

    def __len__(self):
        return len(self.action_list)
    
    def __getitem__(self, index):    
        action=self.action_list[index]
        total_step=action['Action_Len']
        predict_step_number=total_step-2
        Start_Message=action['Action_Label'][0]
        Goal_Message=action['Action_Label'][total_step-1]
        Task_decription = action['Target_Label']
        #Predicted='[sos]'
        Predict=""
        for i in range(1,total_step-1):
            Predict=Predict+action['Action_Label'][i]+'.'
        Predict = Predict + action['Action_Label'][0] + '.' + action['Action_Label'][total_step-1] + '.'
        if self.prompt_mode == 0:
            Prompt=Add_Prompt_0(predict_step_number,Start_Message,Goal_Message)
        elif self.prompt_mode == 1:
            Prompt=Add_Prompt_1(predict_step_number,Start_Message,Goal_Message)
        elif self.prompt_mode == 2:
            Prompt=Add_Prompt_for_Image_Bart(Task_decription,predict_step_number,Start_Message,Goal_Message)
        return Prompt,Predict

class coin_text_dataset_all_123(Dataset):
    def __init__(self, filename , mode = 0):       
        self.filename=filename 
        with open(self.filename,'r',encoding='utf-8') as file:
             self.action_list=json.load(file)
        self.prompt_mode = mode

    def __len__(self):
        return len(self.action_list)
    
    def __getitem__(self, index):    
        action=self.action_list[index]
        total_step=action['Action_Len']
        predict_step_number=total_step-2
        Start_Message=action['Action_Label'][0]
        Goal_Message=action['Action_Label'][total_step-1]
        Task_decription = action['Target_Label']
        #Predicted='[sos]'
        Predict=""
        for i in range(1,total_step-1):
            Predict=Predict+action['Action_Label'][i]+'.'
        Predict = action['Action_Label'][0] + '.' + Predict + action['Action_Label'][total_step-1] + '.'
        if self.prompt_mode == 0:
            Prompt=Add_Prompt_0(predict_step_number,Start_Message,Goal_Message)
        elif self.prompt_mode == 1:
            Prompt=Add_Prompt_1(predict_step_number,Start_Message,Goal_Message)
        elif self.prompt_mode == 2:
            Prompt=Add_Prompt_for_Image_Bart(Task_decription,predict_step_number,Start_Message,Goal_Message)
        return Prompt,Predict
    
class prompt_with_captioning_dataset(Dataset):
    def __init__(self,action_file,caption_file,mode = 0):       
        with open(action_file,'r',encoding='utf-8') as file:
            self.action_list=json.load(file)
        with open(caption_file,'r',encoding='utf-8') as file:
            self.caption_list=json.load(file)
        self.prompt_mode = mode

    def __len__(self):
        return min(len(self.action_list),len(self.caption_list))
    
    def __getitem__(self, index):    
        action=self.action_list[index]
        caption=self.caption_list[index]
        start_caption = caption['start_caption']
        end_caption = caption['end_caption']
        total_step=action['Action_Len']
        predict_step_number=total_step-2
        Start_Message=action['Action_Label'][0]
        Goal_Message=action['Action_Label'][total_step-1]
        Task_decription = action['Target_Label']
        #Predicted='[sos]'
        Predict=""
        for i in range(1,total_step-1):
            Predict=Predict+action['Action_Label'][i]+'.'
        Predict = action['Action_Label'][0] + '.' + Predict + action['Action_Label'][total_step-1] + '.'
        if self.prompt_mode == 0:
            Prompt=Add_Prompt_0(predict_step_number,Start_Message,Goal_Message)
        elif self.prompt_mode == 1:
            Prompt=Add_Prompt_1(predict_step_number,Start_Message,Goal_Message)
        elif self.prompt_mode == 2:
            Prompt=Add_Prompt_for_Image_Bart(Task_decription,predict_step_number,Start_Message,Goal_Message)
        elif self.prompt_mode == 3:
            Prompt=Add_Prompt_Task_description(Task_decription,start_caption,end_caption,predict_step_number+2)
        return Prompt,Predict
    