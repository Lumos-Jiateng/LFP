#Prepare for the train/test dataset structure.
#The coin_train.json contains all the training samples while the coin_test.json contains all the test samples.

#This generate_json_1 file aims to generate predictions for all the intermediate steps
#Originally, train with start + end + prompt -> get start step + end step
#New setting, train with start + end + prompt -> get all the steps in between. (T = 3)

import json
import itertools as it


################## construct the train set here########################
'''
different construtions:
    -- train with single image (start frame + end frame)
    -- train with multiple frames (averaging all the frames)

    -- train with Mixed step number / Train with all the possible steps (Say that conditional inference works)
    -- train with Similar Step range (wait)

'''
################## construct the train set here########################

#Training samples, for each train video of lenth K:  generate samples with only one / two . (seperately)
#We no longer use the short materials to train since they are relatively useless                                

def write_train_json_file(training_type_1,training_type_2,interval,Train_json_path,image_number=2):
    Train_image_root = YOUR_IMAGE_ROOT_PATH
    assert training_type_1 in ['s','m','s0'] 
    assert training_type_2 in ['s','m','s0'] 
    Train_json_file = YOUR_JSON_FILE_PATH
    with open(Train_json_file,'r',encoding='utf-8') as file:
        Train_data = json.load(file)
    Train_data = Train_data['database']

    Train_data_json = []

    for youtube_id in Train_data.keys():
        data = Train_data[youtube_id]
        step_lenth = len(data['annotation'])
        for start in range(step_lenth):
            if start + interval +1 >= step_lenth:
                break
            else:
                index = start+interval+1
                temp_dict = {}
                temp_dict['task_discription'] = data['class']
                temp_dict['interval'] = interval
                temp_dict['start_caption'] = data['annotation'][start]['label']
                temp_dict['end_caption'] = data['annotation'][index]['label']
                temp_dict['start_image_path'] = Train_image_root + str(data['recipe_type']) + '/' + youtube_id + '/' + 'image_'+ training_type_1 +'_' + str(start) + '.jpg'
                temp_dict['end_image_path'] = Train_image_root + str(data['recipe_type']) + '/' + youtube_id + '/' + 'image_'+ training_type_2 +'_' + str(index) + '.jpg'
                if interval >= 1 and image_number>=3:
                    temp_dict['interval_step_1'] =data['annotation'][start+1]['label']
                    temp_dict['interval_1_image_path'] = Train_image_root + str(data['recipe_type']) + '/' + youtube_id + '/' + 'image_' + training_type_1 + '_' + str(start+1) + '.jpg'
                if interval >= 2 and image_number>=4:
                    temp_dict['interval_step_2'] =data['annotation'][start+2]['label']
                    temp_dict['interval_2_image_path'] = Train_image_root + str(data['recipe_type']) + '/' + youtube_id + '/' + 'image_' + training_type_1 + '_' + str(start+2) + '.jpg'
                if interval >= 3 and image_number>=5:
                    temp_dict['interval_step_3'] =data['annotation'][start+3]['label']
                    temp_dict['interval_3_image_path'] = Train_image_root + str(data['recipe_type']) + '/' + youtube_id + '/' + 'image_' + training_type_1 + '_' + str(start+3) + '.jpg'
                Train_data_json.append(temp_dict)  

    with open(Train_json_path,'w',encoding='utf-8') as file:
        json.dump(Train_data_json,file)

    return Train_data_json

################## construct the Test set here########################
'''
different construtions:
    -- test with single image (start frame + end frame)
    -- test with multiple frames (averaging all the frames)

    -- test with step_number = 1
    -- test with step_number = 2
'''
################## construct the Test set here########################
#Testing samples, for each test video of lenth k: generate samples with one / two  steps.

def write_val_json_file(testing_type_1,testing_type_2,interval,Test_json_path):
    Test_image_root = YOUR_IMAGE_ROOT_PATH
    assert testing_type_1 in ['s','m','s0'] 
    assert testing_type_2 in ['s','m','s0'] 
    Test_json_file = YOUR_JSON_FILE_PATH
    with open(Test_json_file,'r',encoding='utf-8') as file:
        Test_data = json.load(file)
    Test_data = Test_data['database']    
    
    Test_data_json = []

    for youtube_id in Test_data.keys():
        data = Test_data[youtube_id]
        step_lenth = len(data['annotation'])
        for start in range(step_lenth):
            if start + interval +1 >= step_lenth:
                break
            else:
                index = start+interval+1
                temp_dict = {}
                temp_dict['task_discription'] = data['class']
                temp_dict['interval'] = interval
                temp_dict['start_caption'] = data['annotation'][start]['label']
                temp_dict['end_caption'] = data['annotation'][index]['label']
                temp_dict['start_image_path'] = Test_image_root + str(data['recipe_type']) + '/' + youtube_id + '/' + 'image_'+ testing_type_1 + '_' + str(start) + '.jpg'
                temp_dict['end_image_path'] = Test_image_root + str(data['recipe_type']) + '/' + youtube_id + '/' + 'image_'+ testing_type_2 + '_'  + str(index) + '.jpg'                
                if interval >= 1:
                    temp_dict['interval_step_1'] =data['annotation'][start+1]['label']
                if interval >= 2:
                    temp_dict['interval_step_2'] =data['annotation'][start+2]['label']
                if interval >= 3:
                    temp_dict['interval_step_3'] =data['annotation'][start+3]['label']
                Test_data_json.append(temp_dict)

    with open(Test_json_path,'w',encoding='utf-8') as file:
        json.dump(Test_data_json,file)   
   
                                 
def write_test_json_file(testing_type_1,testing_type_2,interval,Test_json_path):
    Test_image_root = YOUR_IMAGE_ROOT_PATH
    assert testing_type_1 in ['s','m','s0'] 
    assert testing_type_2 in ['s','m','s0']
    Test_json_file = YOUR_JSON_FILE_PATH
    with open(Test_json_file,'r',encoding='utf-8') as file:
        Test_data = json.load(file)
    Test_data = Test_data['database']    
    
    Test_data_json = []

    for youtube_id in Test_data.keys():
        data = Test_data[youtube_id]
        step_lenth = len(data['annotation'])
        for start in range(step_lenth):
            if start + interval +1 >= step_lenth:
                break
            else:
                index = start+interval+1
                temp_dict = {}
                temp_dict['task_discription'] = data['class']
                temp_dict['interval'] = interval
                temp_dict['start_caption'] = data['annotation'][start]['label']
                temp_dict['end_caption'] = data['annotation'][index]['label']
                temp_dict['start_image_path'] = Test_image_root + str(data['recipe_type']) + '/' + youtube_id + '/' + 'image_'+ testing_type_1 + '_' + str(start) + '.jpg'
                temp_dict['end_image_path'] = Test_image_root + str(data['recipe_type']) + '/' + youtube_id + '/' + 'image_'+ testing_type_2 + '_'  + str(index) + '.jpg'                
                if interval >= 1:
                    temp_dict['interval_step_1'] =data['annotation'][start+1]['label']
                if interval >= 2:
                    temp_dict['interval_step_2'] =data['annotation'][start+2]['label']
                if interval >= 3:
                    temp_dict['interval_step_3'] =data['annotation'][start+3]['label']
                Test_data_json.append(temp_dict)

    with open(Test_json_path,'w',encoding='utf-8') as file:
        json.dump(Test_data_json,file)
    


if __name__ == '__main__':
    # standard results for Coin T=3
    write_train_json_file('m','m',1,'/nfs4-p1/ljt/github/coin_preprocessing_new/Json_for_dataset_final/Multi_Image/one_train.json',T)
    write_val_json_file('m','m',1,'/nfs4-p1/ljt/github/coin_preprocessing_new/Json_for_dataset_final/Multi_Image/one_val.json')
    write_test_json_file('m','m',1,'/nfs4-p1/ljt/github/coin_preprocessing_new/Json_for_dataset_final/Multi_Image/one_test.json')
    
    write_train_json_file('s','s0',1,'/nfs4-p1/ljt/github/coin_preprocessing_new/Json_for_dataset_final/Single_Image/one_train.json',T)
    write_val_json_file('s','s0',1,'/nfs4-p1/ljt/github/coin_preprocessing_new/Json_for_dataset_final/Single_Image/one_val.json')
    write_test_json_file('s','s0',1,'/nfs4-p1/ljt/github/coin_preprocessing_new/Json_for_dataset_final/Single_Image/one_test.json')