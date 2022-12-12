from torch.utils.data import Dataset
import torch
import numpy as np
import random
import cv2
import json
import os
from data.crosstask_steps import Crosstask_steps,task_id_dict,task_steps_dict,inverse_dict
from data.crosstask_steps import step_dictionary,step_dictionary_inverse

class Train_CrossTask_Retrieve_Dataset_with_Taskname_Double(Dataset):
    def __init__(self,json_file,feature_path):
        with open(json_file,'r',encoding='utf-8') as f:
            self.data_list = json.load(f)
        self.feature_path = feature_path

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_piece = self.data_list[index]
        video_name = self.data_list[index]['video_name']
        # get to ff info
        ff_start = self.data_list[index]['ff_start_number']
        ff_end = self.data_list[index]['ff_end_number']
        X = torch.tensor(np.load(os.path.join(self.feature_path,video_name+'.npy')), dtype=torch.float)
        #print(X.shape,ff_start,ff_end)
        ff_start_img_feature = X[ff_start-1:ff_start+2]
        ff_end_img_feature = X[ff_end-2]
        # get to lf info
        lf_start = self.data_list[index]['lf_start_number']
        lf_end = self.data_list[index]['lf_end_number']
        #Y = torch.tensor(np.load(os.path.join(self.feature_path,video_name+'.npy')), dtype=torch.float)
        #print(X.shape,lf_start,lf_end)
        lf_start_img_feature = X[lf_start-1]
        lf_end_img_feature = X[lf_end-3:lf_end]

        if(data_piece['prediction_range']) == 0:
            text_description = 'no step'
            text_string = 'The task is ' + data_piece['Target_Label'] + ' and there is ' + text_description + ' in between .'
        elif(data_piece['prediction_range']) == 1:
            text_description = 'only one step'
            text_string = 'The task is ' + data_piece['Target_Label'] + ' and there is ' + text_description + ' in between .'
        elif(data_piece['prediction_range']) == 2:
            text_description = 'two steps'
            text_string = 'The task is ' + data_piece['Target_Label'] + ' and there are ' + text_description + ' in between .'
        
        if ff_start_img_feature.shape[0]!=3 or lf_end_img_feature.shape[0]!=3:
            print('error detected') 
            return self.__getitem__(0)

        return ff_start_img_feature,ff_end_img_feature,lf_start_img_feature,lf_end_img_feature,step_dictionary[data_piece['ff_Action_Label']],step_dictionary[data_piece['lf_Action_Label']],text_string

class Eval_CrossTask_Retrieve_Dataset_with_Taskname(Dataset):
    def __init__(self,json_file,feature_path):
        with open(json_file,'r',encoding='utf-8') as f:
            self.data_list = json.load(f)
        self.feature_path = feature_path
        self.crosstask_steps = Crosstask_steps
        self.description2id = inverse_dict(task_id_dict)
        self.id2list = task_steps_dict
        assert(len(self.crosstask_steps) == 105)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_piece = self.data_list[index]
        video_name = self.data_list[index]['video_name']
        start_s = self.data_list[index]['start_times'][0]
        #end_s = self.data_list[index]['end_number'][0]
        #start_g = self.data_list[index]['start_number'][-1]
        end_g = self.data_list[index]['end_times'][-1]
        X = torch.tensor(np.load(os.path.join(self.feature_path,video_name+'.npy')), dtype=torch.float)
        #if end_g >= X.shape[0]-1:
        #    print(video_name)
        start_img_feature = X[start_s-1:start_s+2]
        end_img_feature = X[end_g-3:end_g] # in case of exceeding range
        
        step_number = len(data_piece['Action_Label']) - 2
        if step_number == 2:
            text_string = 'The task is ' + data_piece['Target_Label'] + ' and there are ' + 'two steps' + ' in between'
        elif step_number == 1:
            text_string = 'The task is ' + data_piece['Target_Label'] + ' and there is only ' + 'one step' + ' in between'
        
        if start_img_feature.shape[0]!=3 or end_img_feature.shape[0]!=3:
            print('val error detected') 
            return self.__getitem__(0)
        # return format : description + feature + feature + label_ff + label_lf
        return text_string,start_img_feature,end_img_feature,step_dictionary[data_piece['Action_Label'][0]],step_dictionary[data_piece['Action_Label'][-1]]
        

if __name__ == '__main__':
    test_data = Eval_CrossTask_Retrieve_Dataset_with_Taskname('/nfs4-p1/ljt/Code/BRE/crosstask_dataset/CrossTask/json_for_LM/cross_task_test_step_3.json','/nfs4-p1/ljt/Code/BRE/crosstask_dataset/CrossTask/crosstask_features/')
    print(test_data[0])