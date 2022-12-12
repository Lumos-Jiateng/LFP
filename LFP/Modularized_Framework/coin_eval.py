import argparse
from email.mime import image
import ruamel.yaml as yaml
from pathlib import Path
import os
import torchvision

from transformers import StoppingCriteria
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import datetime
import random
import numpy as np
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from data import create_dataset,create_loader
import json
from models.coin_align import load_coin_align_with_blip
from utils import cosine_lr_schedule
from models.Bartbased_coin_finetune import text_baseline
from utils import Add_Prompt_0,Add_Prompt_1,Number_to_String,Add_Prompt_for_Image_Bart
from data.coin180_custom import step_label_180,label_step_180,step_list_180
from sentence_transformers import SentenceTransformer

def content_convert(input_list,sentence_bert_model,target_matrix):
    new_list = []
    embeddings = sentence_bert_model.encode(input_list)
    similarity = np.dot(embeddings,target_matrix.transpose())
    for i in range(similarity.shape[0]):
        new_list.append(label_step_180[np.argmax(similarity[i])])
    #print(f'original list is {input_list}')
    #print(f'new list is {new_list}')
    return new_list

def Eval_pipeline_model (model,langauge_model,data_loader,device,write_path,step_number,sentence_bert_model,if_align = False):
    
    model.eval()
    langauge_model.eval()
    target_matrix = sentence_bert_model.encode(step_list_180)
    # define metrics
    Sr = 0
    mAcc = 0
    MIoU = 0

    sample_number = 0
    SR_number = 0
    acc_number = 0
    IoU_number = 0
    Example_write_data = []
    if step_number == 1:        
        for i,(task_description,ff_image,lf_image,ff_label,step_1_label,lf_label) in enumerate (data_loader):
            ff_image = ff_image.to(device,non_blocking=True) 
            lf_image = lf_image.to(device,non_blocking=True)
            ff_feature = model.visual_encoder(ff_image) 
            lf_feature = model.visual_encoder(lf_image) 

            ff_start_atts = torch.ones(ff_feature.size()[:-1],dtype=torch.long).to(ff_feature.device) 
            lf_end_atts = torch.ones(lf_feature.size()[:-1],dtype=torch.long).to(ff_feature.device) 
        
            text = model.tokenizer(task_description, padding='max_length', return_tensors="pt").to(ff_feature.device)
            output = model.text_encoder(text.input_ids, 
                                        attention_mask = text.attention_mask, 
                                        encoder_hidden_states = [ff_feature,lf_feature],
                                        encoder_attention_mask = [ff_start_atts,lf_end_atts],        
                                        return_dict = True,
                                        ) 
            hidden_state = output.last_hidden_state[:,0,:]         
            ff_prediction = model.cls_head_for_ff(hidden_state)
            lf_prediction = model.cls_head_for_lf(hidden_state) # probability values
            ff_indexes = torch.topk(ff_prediction,3,1)[1] 
            lf_indexes = torch.topk(lf_prediction,3,1)[1] # (bz,3)
            for j in range(ff_indexes.shape[0]):
                # Bart inference with previous results.
                task_name = task_description[j].split(' ')[3]
                Prompt = Add_Prompt_0(step_number,label_step_180[ff_indexes[j][0].item()],label_step_180[lf_indexes[j][0].item()])
                #print(Prompt)
                Inputs=langauge_model.pretrained_tokenizer(Prompt, truncation=True, padding=True,return_tensors='pt')
                summary_ids = langauge_model.pretrained_model.generate(Inputs['input_ids'].to(device,non_blocking=True), 
                                                      num_beams=4, max_length=30, early_stopping=True)
                current_pred=[langauge_model.pretrained_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0]
                current_pred_list=current_pred.split('.')
                if(current_pred_list[-1] == ''):
                    current_pred_list.pop()
                    
                # add a line of aligning if align = True
                if if_align == True:
                    current_pred_list = content_convert(current_pred_list,sentence_bert_model,target_matrix)
                sample_number = sample_number + 1
                
                # labels / indexes are tensor numbers while the later are texts
                if(ff_indexes[j][0] == ff_label[j] and lf_indexes[j][0] == lf_label[j] and current_pred_list[0] == step_1_label[j]):
                    SR_number = SR_number + 1
                if(ff_indexes[j][0] == ff_label[j]):
                    acc_number = acc_number + 1
                if(lf_indexes[j][0] == lf_label[j]):
                    acc_number = acc_number + 1
                if(current_pred_list[0] == step_1_label[j]):
                    acc_number = acc_number + 1
                if ff_indexes[j][0] == ff_label[j] or ff_indexes[j][0] == lf_label[j] or ff_indexes[j][0] == step_1_label[j]:
                    IoU_number = IoU_number + 1
                if lf_indexes[j][0] == ff_label[j] or lf_indexes[j][0] == lf_label[j] or lf_indexes[j][0] == step_1_label[j]:
                    IoU_number = IoU_number + 1
                if current_pred_list[0] == ff_label[j] or current_pred_list[0] == lf_label[j] or current_pred_list[0] == step_1_label[j]:
                    IoU_number = IoU_number + 1
                temp_dict = {}
                temp_dict['real_label'] = [label_step_180[ff_label[j].item()],step_1_label[j],label_step_180[lf_label[j].item()]]
                temp_dict['predicted_steps'] = [label_step_180[ff_indexes[j][0].item()],current_pred_list[0],label_step_180[lf_indexes[j][0].item()]]
                Example_write_data.append(temp_dict)
                print(SR_number / sample_number)
        
        Sr = 1.0* SR_number / sample_number
        mAcc = 1.0 * acc_number / sample_number / 3
        MIoU = 1.0 * IoU_number / sample_number / 3
        with open(write_path,'w') as f:
            json.dump(Example_write_data,f)
        return {'SR':Sr, 'mAcc':mAcc,'MIoU':MIoU}

    
    if step_number == 2:        
        for i,(task_description,ff_image,lf_image,ff_label,step_1_label,step_2_label,lf_label) in enumerate (data_loader):
            ff_image = ff_image.to(device,non_blocking=True) 
            lf_image = lf_image.to(device,non_blocking=True)
            ff_feature = model.visual_encoder(ff_image) 
            lf_feature = model.visual_encoder(lf_image) 

            ff_start_atts = torch.ones(ff_feature.size()[:-1],dtype=torch.long).to(ff_feature.device) 
            lf_end_atts = torch.ones(lf_feature.size()[:-1],dtype=torch.long).to(ff_feature.device) 
        
            text = model.tokenizer(task_description, padding='max_length', return_tensors="pt").to(ff_feature.device)
            output = model.text_encoder(text.input_ids, 
                                        attention_mask = text.attention_mask, 
                                        encoder_hidden_states = [ff_feature,lf_feature],
                                        encoder_attention_mask = [ff_start_atts,lf_end_atts],        
                                        return_dict = True,
                                        ) 
            hidden_state = output.last_hidden_state[:,0,:]         
            ff_prediction = model.cls_head_for_ff(hidden_state)
            lf_prediction = model.cls_head_for_lf(hidden_state) # probability values
            ff_indexes = torch.topk(ff_prediction,3,1)[1] 
            lf_indexes = torch.topk(lf_prediction,3,1)[1] # (bz,3)
            for j in range(ff_indexes.shape[0]):
                # Bart inference with previous results.
                task_name = task_description[j].split(' ')[3]
                #print(task_name)
                
                Prompt = Add_Prompt_for_Image_Bart(task_name,step_number,label_step_180[ff_indexes[j][0].item()],label_step_180[lf_indexes[j][0].item()])
                Inputs=langauge_model.pretrained_tokenizer(Prompt, truncation=True, padding=True,return_tensors='pt')
                summary_ids = langauge_model.pretrained_model.generate(Inputs['input_ids'].to(device,non_blocking=True), 
                                                      num_beams=4, max_length=30, early_stopping=True)
                current_pred=[langauge_model.pretrained_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0]
                current_pred_list=current_pred.split('.')
                if(current_pred_list[-1] == ''):
                    current_pred_list.pop()
                
                # adjust current_pred_list here if Use Sentence Bert to align
                if if_align == True:
                    current_pred_list = content_convert(current_pred_list,sentence_bert_model,target_matrix)
                # ？？？？ what the fuck is here ！
                #sample_number = sample_number + 1
                
                sample_number = sample_number + 1
                if(ff_indexes[j][0] == ff_label[j] and lf_indexes[j][0] == lf_label[j] and current_pred_list[0] == step_1_label[j] and current_pred_list[1] == step_2_label[j]):
                    SR_number = SR_number + 1
                if(ff_indexes[j][0] == ff_label[j]):
                    acc_number = acc_number + 1
                if(lf_indexes[j][0] == lf_label[j]):
                    acc_number = acc_number + 1
                if(current_pred_list[0] == step_1_label[j]):
                    acc_number = acc_number + 1
                if(current_pred_list[1] == step_2_label[j]):
                    acc_number = acc_number + 1
                if ff_indexes[j][0] == ff_label[j] or ff_indexes[j][0] == lf_label[j] or ff_indexes[j][0] == step_1_label[j] or ff_indexes[j][0] == step_2_label[j]:
                    IoU_number = IoU_number + 1
                if lf_indexes[j][0] == ff_label[j] or lf_indexes[j][0] == lf_label[j] or lf_indexes[j][0] == step_1_label[j] or lf_indexes[j][0] == step_2_label[j]:
                    IoU_number = IoU_number + 1
                if current_pred_list[0] == ff_label[j] or current_pred_list[0] == lf_label[j] or current_pred_list[0] == step_1_label[j] or current_pred_list[0] == step_2_label[j]:
                    IoU_number = IoU_number + 1
                if current_pred_list[1] == ff_label[j] or current_pred_list[1] == lf_label[j] or current_pred_list[1] == step_1_label[j] or current_pred_list[1] == step_2_label[j]:
                    IoU_number = IoU_number + 1
                temp_dict = {}
                temp_dict['real_label'] = [label_step_180[ff_label[j].item()],step_1_label[j],step_2_label[j],label_step_180[lf_label[j].item()]]
                temp_dict['predicted_steps'] = [label_step_180[ff_indexes[j][0].item()],current_pred_list[0],current_pred_list[1],label_step_180[lf_indexes[j][0].item()]]
                Example_write_data.append(temp_dict)
                print(SR_number / sample_number)

        Sr = 1.0* SR_number / sample_number
        mAcc = 1.0 * acc_number / sample_number / 4
        MIoU = 1.0 * IoU_number / sample_number / 4
        with open(write_path,'w') as f:
            json.dump(Example_write_data,f)
        return {'SR':Sr, 'mAcc':mAcc,'MIoU':MIoU}    

    if step_number == 3:        
        try:
            for i,(task_description,ff_image,lf_image,ff_label,step_1_label,step_2_label,step_3_label,lf_label) in enumerate (data_loader):
                ff_image = ff_image.to(device,non_blocking=True) 
                lf_image = lf_image.to(device,non_blocking=True)
                ff_feature = model.visual_encoder(ff_image) 
                lf_feature = model.visual_encoder(lf_image) 

                ff_start_atts = torch.ones(ff_feature.size()[:-1],dtype=torch.long).to(ff_feature.device) 
                lf_end_atts = torch.ones(lf_feature.size()[:-1],dtype=torch.long).to(ff_feature.device) 
        
                text = model.tokenizer(task_description, padding='max_length', return_tensors="pt").to(ff_feature.device)
                output = model.text_encoder(text.input_ids, 
                                        attention_mask = text.attention_mask, 
                                        encoder_hidden_states = [ff_feature,lf_feature],
                                        encoder_attention_mask = [ff_start_atts,lf_end_atts],        
                                        return_dict = True,
                                        ) 
                hidden_state = output.last_hidden_state[:,0,:]         
                ff_prediction = model.cls_head_for_ff(hidden_state)
                lf_prediction = model.cls_head_for_lf(hidden_state) # probability values
                ff_indexes = torch.topk(ff_prediction,3,1)[1] 
                lf_indexes = torch.topk(lf_prediction,3,1)[1] # (bz,3)
                for j in range(ff_indexes.shape[0]):
                 # Bart inference with previous results.
                    task_name = task_description[j].split(' ')[3]
                #print(task_name)
                
                    Prompt = Add_Prompt_for_Image_Bart(task_name,step_number,label_step_180[ff_indexes[j][0].item()],label_step_180[lf_indexes[j][0].item()])
                    Inputs=langauge_model.pretrained_tokenizer(Prompt, truncation=True, padding=True,return_tensors='pt')
                    summary_ids = langauge_model.pretrained_model.generate(Inputs['input_ids'].to(device,non_blocking=True), 
                                                          num_beams=4, max_length=30, early_stopping=True)
                    current_pred=[langauge_model.pretrained_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0]
                    current_pred_list=current_pred.split('.')
                    if(current_pred_list[-1] == ''):
                        current_pred_list.pop()
                
                # adjust current_pred_list here if Use Sentence Bert to align
                    if if_align == True:
                        current_pred_list = content_convert(current_pred_list,sentence_bert_model,target_matrix)
                # ？？？？ what the fuck is here ！
                #sample_number = sample_number + 1
                
                    sample_number = sample_number + 1
                    if(ff_indexes[j][0] == ff_label[j] and lf_indexes[j][0] == lf_label[j] and current_pred_list[0] == step_1_label[j] and current_pred_list[1] == step_2_label[j] and current_pred_list[2] == step_3_label[j]):
                        SR_number = SR_number + 1
                    if(ff_indexes[j][0] == ff_label[j]):
                        acc_number = acc_number + 1
                    if(lf_indexes[j][0] == lf_label[j]):
                        acc_number = acc_number + 1
                    if(current_pred_list[0] == step_1_label[j]):
                        acc_number = acc_number + 1
                    if(current_pred_list[1] == step_2_label[j]):
                        acc_number = acc_number + 1
                    if(current_pred_list[2] == step_3_label[j]):
                        acc_number = acc_number + 1
                    if ff_indexes[j][0] == ff_label[j] or ff_indexes[j][0] == lf_label[j] or ff_indexes[j][0] == step_1_label[j] or ff_indexes[j][0] == step_2_label[j] or ff_indexes[j][0] == step_3_label[j]:
                        IoU_number = IoU_number + 1
                    if lf_indexes[j][0] == ff_label[j] or lf_indexes[j][0] == lf_label[j] or lf_indexes[j][0] == step_1_label[j] or lf_indexes[j][0] == step_2_label[j] or lf_indexes[j][0] == step_3_label[j]:
                        IoU_number = IoU_number + 1
                    if current_pred_list[0] == ff_label[j] or current_pred_list[0] == lf_label[j] or current_pred_list[0] == step_1_label[j] or current_pred_list[0] == step_2_label[j] or current_pred_list[0] == step_3_label[j]:
                        IoU_number = IoU_number + 1
                    if current_pred_list[1] == ff_label[j] or current_pred_list[1] == lf_label[j] or current_pred_list[1] == step_1_label[j] or current_pred_list[1] == step_2_label[j] or current_pred_list[1] == step_3_label[j]:
                        IoU_number = IoU_number + 1
                    if current_pred_list[2] == ff_label[j] or current_pred_list[2] == lf_label[j] or current_pred_list[2] == step_1_label[j] or current_pred_list[2] == step_2_label[j] or current_pred_list[2] == step_3_label[j]:
                        IoU_number = IoU_number + 1
                    temp_dict = {}
                    temp_dict['real_label'] = [label_step_180[ff_label[j].item()],step_1_label[j],step_2_label[j],step_3_label[j],label_step_180[lf_label[j].item()]]
                    temp_dict['predicted_steps'] = [label_step_180[ff_indexes[j][0].item()],current_pred_list[0],current_pred_list[1],current_pred_list[2],label_step_180[lf_indexes[j][0].item()]]
                    Example_write_data.append(temp_dict)
                    print(SR_number / sample_number)
        
        except BaseException as e:
            print(current_pred_list)
            
        Sr = 1.0* SR_number / sample_number
        mAcc = 1.0 * acc_number / sample_number / 5
        MIoU = 1.0 * IoU_number / sample_number / 5
        with open(write_path,'w') as f:
            json.dump(Example_write_data,f)
        return {'SR':Sr, 'mAcc':mAcc,'MIoU':MIoU}  

def train(model, data_loader, optimizer, epoch, device, config):
    # train
    model.train()  

    for i, (text_description, ff_start,ff_end,ff_action_label,lf_action_label) in enumerate(data_loader):

        ff_start = ff_start.to(device,non_blocking=True)   
        ff_end = ff_end.to(device,non_blocking=True)  
        #lf_start = lf_start.to(device,non_blocking=True)   
        #lf_end = lf_end.to(device,non_blocking=True) 
        ff_action_label =  ff_action_label.to(device,non_blocking=True) 
        lf_action_label =  lf_action_label.to(device,non_blocking=True) 
               
        loss_ff , loss_lf , language_loss_ff , language_loss_lf = model(text_description,ff_start,ff_end,ff_action_label,lf_action_label)    
        loss = loss_ff + loss_lf + language_loss_ff + language_loss_lf
        #loss = loss_ff + loss_lf
        #loss = language_loss_ff + language_loss_lf
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    

        print(f"Averaged stats: epoch: {epoch}  loss : {loss} cls_loss_ff : {loss_ff} cls_loss_lf : {loss_lf}  ")
        print(f"language_loss_ff : {language_loss_ff} language_loss_lf:{language_loss_lf}")
        #print(f"Averaged stats: epoch: {epoch}  loss : {loss} cls_loss : {loss_ff+loss_lf}") 
        #print(f"Averaged stats: epoch: {epoch}  loss : {loss} language_loss : {language_loss_ff + language_loss_lf}") 

def double_retrieval_eval(model, data_loader, device, write_path, step_number): # step_number = T-2
    
    model.eval()    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    start_time = time.time()
    #begin evaluation
    
    # evaluate in unit of batches
    sample_number = 0
    correct_number = 0
    ff_correct_number = 0
    lf_correct_number = 0
    ff_top3 = 0
    lf_top3 = 0
    record_list = []
    for i,(task_description,ff_image,lf_image,ff_label,step_1_capition,lf_label) in enumerate (data_loader):
        ff_image = ff_image.to(device,non_blocking=True) 
        lf_image = lf_image.to(device,non_blocking=True)
        ff_feature = model.visual_encoder(ff_image) 
        lf_feature = model.visual_encoder(lf_image) 
        # change them to valid shape of input
        #ff_feature = torch.unsqueeze(ff_feature,1)
        #lf_feature = torch.unsqueeze(lf_feature,1)
        ff_start_atts = torch.ones(ff_feature.size()[:-1],dtype=torch.long).to(ff_feature.device) 
        lf_end_atts = torch.ones(lf_feature.size()[:-1],dtype=torch.long).to(ff_feature.device) 
        
        text = model.tokenizer(task_description, padding='max_length', return_tensors="pt").to(ff_feature.device)
        output = model.text_encoder(text.input_ids, 
                                    attention_mask = text.attention_mask, 
                                    encoder_hidden_states = [ff_feature,lf_feature],
                                    encoder_attention_mask = [ff_start_atts,lf_end_atts],        
                                    return_dict = True,
                                    ) 
        hidden_state = output.last_hidden_state[:,0,:]         
        ff_prediction = model.cls_head_for_ff(hidden_state)
        lf_prediction = model.cls_head_for_lf(hidden_state) # probability values
        ff_indexes = torch.topk(ff_prediction,3,1)[1] 
        lf_indexes = torch.topk(lf_prediction,3,1)[1] # (bz,3)
        for j in range(ff_indexes.shape[0]):
            temp_record = {}
            sample_number = sample_number + 1
            if ff_indexes[j][0] == ff_label[j] and lf_indexes[j][0] == lf_label[j]:
                correct_number = correct_number + 1
            elif ff_indexes[j][0] == ff_label[j] and lf_indexes[j][0] != lf_label[j]:
                ff_correct_number = ff_correct_number + 1
            elif ff_indexes[j][0] != ff_label[j] and lf_indexes[j][0] == lf_label[j]:
                lf_correct_number = lf_correct_number + 1
            if ff_indexes[j][0] == ff_label[j] or ff_indexes[j][1] == ff_label[j] or ff_indexes[j][2] == ff_label[j]:
                ff_top3 = ff_top3 + 1
            if lf_indexes[j][0] == lf_label[j] or lf_indexes[j][1] == lf_label[j] or lf_indexes[j][2] == lf_label[j]:
                lf_top3 = lf_top3 + 1
            temp_record['text_description'] = task_description[j]
            temp_record['real_label'] = (label_step_180[ff_label[j].item()],label_step_180[lf_label[j].item()])
            temp_record['predicted_label'] = (label_step_180[ff_indexes[j][0].item()],label_step_180[lf_indexes[j][0].item()])
            temp_record['ff_top3_predicted_label'] = (label_step_180[ff_indexes[j][0].item()],label_step_180[ff_indexes[j][1].item()],label_step_180[ff_indexes[j][2].item()])
            temp_record['lf_top3_predicted_label'] = (label_step_180[lf_indexes[j][0].item()],label_step_180[lf_indexes[j][1].item()],label_step_180[lf_indexes[j][2].item()])
            record_list.append(temp_record)
    
    # write infering examples / output
    print(f'overall_acc:{1.0* correct_number/sample_number}')
    print(f'only_ff_acc:{1.0* ff_correct_number/sample_number}')
    print(f'only_lf_acc:{1.0* lf_correct_number/sample_number}')
    with open(write_path,'w') as f:
        json.dump(record_list,f)
    overall_retrieval_acc = 1.0* correct_number / sample_number
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))

    return {'overall_acc':overall_retrieval_acc,
            'only_ff_acc':1.0* ff_correct_number/sample_number,
            'only_lf_acc':1.0* lf_correct_number/sample_number,
            'ff_top3_acc':1.0* ff_top3/sample_number,
            'lf_top3_acc':1.0* lf_top3/sample_number }

def main(args, config):  
    
    device = torch.device(args.device)

    #### Dataset #### 
    print("Creating coin_text dataset")
    train_dataset, val_dataset,test_dataset = create_dataset(config)  
    
    print(train_dataset.__len__(),val_dataset.__len__(),test_dataset.__len__())

    samplers = [None, None, None]
    
    train_loader,val_loader,test_loader = create_loader([train_dataset,val_dataset,test_dataset],samplers,
                                                batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                                num_workers=[0,0,0],
                                                is_trains=[True, False,False], 
                                                collate_fns=[None,None,None])   
    if config['step_number'] == 'one':
        config['step_number'] = 1
    elif config['step_number'] == 'two':
        config['step_number'] = 2
    elif config['step_number'] == 'three':
        config['step_number'] = 3

    if(config['step_number'] == 1):
        print('T = 3')
        for i, (description,ff_start,ff_end,start_caption,end_caption) in enumerate(train_loader):
            if i == 0 :
                print(description)
                print(ff_start.shape,ff_end.shape)
            else:
                break
        for i, (description,ff_start,ff_end,start_caption,step_1_caption,end_caption) in enumerate(test_loader):
            if i == 0 :
                print(description)
                print(ff_start.shape,ff_end.shape)
            else:
                break
    elif(config['step_number'] == 2):
        print('T = 4')
        for i, (description,ff_start,ff_end,start_caption,step_1_caption,step_2_caption,end_caption) in enumerate(test_loader):
            if i == 0 :
                print(description)
                print(ff_start.shape,ff_end.shape)
            else:
                break
    
    #### Model #### 
    print("Creating model")
    model = load_coin_align_with_blip(pretrained=config['pretrained'])
    
    #load pretrained checkpoint
    if args.evaluate:
        checkpoint_path = config['double_predict_path']
        ckpt = torch.load(checkpoint_path)
        state_dict = ckpt['model']
        msg = model.load_state_dict(state_dict,strict=False)
        print(msg)           
        model = model.to(device)
               
    model = model.to(device)   
    
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay']) 
    
    best = 0
    best_epoch = 0
    print("Start training")
    start_time = time.time()
    Best_SR = 0

    result_list = []
    
    sentence_bert_model = SentenceTransformer('all-MiniLM-L6-v2')
    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
            train_stats = train(model, train_loader, optimizer, epoch, device, config) 

        #print('test on training set')
        #Train_Retrieve_SR = double_retrieval_eval_withrange(model, test_loader, 15, device, './output_coin/train_prediction.json' ,config['step_number'])

        print('test_result is:')
        #Test_Retrieve_SR = double_retrieval_eval(model, test_loader, device, args.write_path,config['step_number']) 
        
        #record_current_result
        temp_dict = {}
        #temp_dict['test_result'] = Test_Retrieve_SR
        '''
        if Test_Retrieve_SR['overall_acc']>Best_SR:
            Best_SR = Test_Retrieve_SR['overall_acc']
            save_obj = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            dataset_name = config['dataset']
            step_number = config['step_number']
            torch.save(save_obj, os.path.join(args.output_dir, f'checkpoint_{dataset_name}_epoch_{epoch}_T_{step_number}_retrieval.pth')) 
        '''
        if args.eval_pipeline == True:
            if epoch == 0:
                Language_finetuned_model= text_baseline(config=config, pretrianed_model=args.pretrained_model)    
                print('load pretrained language model')
                if config['step_number'] == 1:
                    checkpoint_path = config['lm_checkpoint_path_step_1']
                elif config['step_number'] == 2:
                    checkpoint_path = config['lm_checkpoint_path_step_2']
                elif config['step_number'] == 3:
                    checkpoint_path = config['lm_checkpoint_path_step_3']
                print(f'load_checkpoint_from{checkpoint_path}')
                ckpt = torch.load(checkpoint_path)
                state_dict = ckpt['model']
                msg = Language_finetuned_model.load_state_dict(state_dict,strict=False)
                print(msg)           
                Language_finetuned_model = Language_finetuned_model.to(device) 
            
            Overall_result = Eval_pipeline_model(model,Language_finetuned_model,test_loader,device,args.write_path_all,config['step_number'],sentence_bert_model)
            temp_dict['test_result_pipeline_no_align'] = Overall_result
            Overall_result_a = Eval_pipeline_model(model,Language_finetuned_model,test_loader,device,args.write_path_all,config['step_number'],sentence_bert_model,True)
            temp_dict['test_result_pipeline_align'] = Overall_result_a
            
        result_list.append(temp_dict)
        with open(args.result_path,'w') as f:
            json.dump(result_list,f)

        if args.evaluate:
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='./configs/coin_eval_m2.yaml')
    parser.add_argument('--output_dir', default='./output/coin_eval_m2/text_retrieval_coin/')        
    parser.add_argument('--checkpoint', default='')   
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--write_path', default='./output/coin_eval_2/retrieval_prediction.json')
    parser.add_argument('--write_path_all', default='./output/coin_eval_m2/all_prediction.json')
    parser.add_argument('--result_path', default='./output/coin_eval_m2/results.json')
    parser.add_argument('--eval_pipeline',default=True)
    parser.add_argument('--pretrained_model', default='BartForConditionalGerneration')
    
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    #copy config file into the output directory.
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)