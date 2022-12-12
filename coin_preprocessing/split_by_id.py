# function descriptions
# using range 150 - 179 COIN dataset to generate :
# 1. coin images for training / validating/ testing . 
# 2. corresponding text descriptions in json file to load dataset .
# 3. generate 'fixed language annotation with Clip text encoder' .  

# Target : using Blip pretrained model to init ---- train a double conditional retrieval modal.

# Blip model base : https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth

import json

Original_json_file = '/nfs4-p1/ljt/Code/BRE/coin_preprocessing/COIN.json'
coin_test_json = '/nfs4-p1/ljt/github/coin_preprocessing_new/Part_Json/test_coin.json'
coin_val_json = '/nfs4-p1/ljt/github/coin_preprocessing_new/Part_Json/val_coin.json'
coin_train_json = '/nfs4-p1/ljt/github/coin_preprocessing_new/Part_Json/train_coin.json'

with open(Original_json_file,'r',encoding='utf-8') as file:
    Overall_annotation = json.load(file)
Overall_Video_info = Overall_annotation['database']

dict_train = {}
dict_val = {}
dict_test = {}

video_id_dict = {}
# First, keep a dictionary of recipe_types, each of them being a youtube id
for youtube_id in Overall_annotation['database']:
    Video_info = Overall_Video_info[youtube_id]
    if Video_info['recipe_type'] in video_id_dict.keys():
        video_id_dict[Video_info['recipe_type']].append(youtube_id)
    else:
        video_id_dict[Video_info['recipe_type']] = []
        video_id_dict[Video_info['recipe_type']].append(youtube_id)

train_id_list = []
val_id_list = []
test_id_list = []

for key in video_id_dict.keys():
    id_list = video_id_dict[key]
    training_lenth = int(len(id_list)*0.56) + 1 
    val_lenth = int(len(id_list)*0.14)
    test_lenth = len(id_list) -training_lenth - val_lenth
    print(training_lenth,val_lenth),test_lenth
    for k in range(training_lenth):
        train_id_list.append(id_list[k])
    for k in range(training_lenth,training_lenth+val_lenth):
        val_id_list.append(id_list[k])
    for k in range(training_lenth+val_lenth,training_lenth+val_lenth+test_lenth):
        test_id_list.append(id_list[k])

print(len(train_id_list),len(val_id_list),len(test_id_list))

for youtube_id in Overall_annotation['database']:
    Video_info = Overall_Video_info[youtube_id]
    if youtube_id in train_id_list:
        Video_info['subset'] = 'Training'
        dict_train[youtube_id] = Video_info
    elif youtube_id in val_id_list:
        Video_info['subset'] = 'validating'
        dict_val[youtube_id] = Video_info
    elif youtube_id in test_id_list:
        Video_info['subset'] = 'testing'
        dict_test[youtube_id] = Video_info
      
written_train = {}
written_train['database'] = dict_train
written_val = {}
written_val['database'] = dict_val  
written_test = {}
written_test['database'] = dict_test

with open(coin_train_json,'w',encoding='utf-8') as file:
    json.dump(written_train,file)
    
with open(coin_val_json,'w',encoding='utf-8') as file:
    json.dump(written_val,file)

with open(coin_test_json,'w',encoding='utf-8') as file:
    json.dump(written_test,file)
