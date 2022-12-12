import argparse
from email.mime import image
import ruamel.yaml as yaml
from pathlib import Path
import os

from transformers import StoppingCriteria
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
from data.coin180_custom import step_list_180,label_step_180

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
    for i,(task_description,ff_image,lf_image,ff_label,lf_label) in enumerate (data_loader):
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
            #temp_record['ff_top3_predicted_label'] = (label_step_180[ff_indexes[j][0].item()],label_step_180[ff_indexes[j][1].item()],label_step_180[ff_indexes[j][2].item()])
            #temp_record['lf_top3_predicted_label'] = (label_step_180[lf_indexes[j][0].item()],label_step_180[lf_indexes[j][1].item()],label_step_180[lf_indexes[j][2].item()])
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
    
    print(train_dataset.__len__())
    print(val_dataset.__len__())
    print(test_dataset.__len__())  

    samplers = [None, None, None]
    
    train_loader,val_loader,test_loader = create_loader([train_dataset,val_dataset,test_dataset],samplers,
                                                batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                                num_workers=[4,4,4],
                                                is_trains=[True, False,False], 
                                                collate_fns=[None,None,None])   
    

    print(train_loader.__len__())
    print(val_loader.__len__())
    print(test_loader.__len__())
    
    for i, (description,ff_start,ff_end,start_caption,end_caption) in enumerate(train_loader):
        if i == 0 :
            print(description)
            print(ff_start.shape,ff_end.shape)
        else:
            break
    #for i, data in enumerate(val_loader):
    #        continue
    #for i, data in enumerate(test_loader):
    #        continue

    #### Model #### 
    print("Creating model")
    model = load_coin_align_with_blip(pretrained=config['pretrained'])
               
    model = model.to(device)   
    
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay']) 
    
    if args.checkpoint:    
        checkpoint_path = args.checkpoint
        print(f'load_checkpoint_from{checkpoint_path}')
        ckpt = torch.load(checkpoint_path)
        state_dict = ckpt['model']
        msg = model.load_state_dict(state_dict,strict=False)
        print(msg)
    
    best = 0
    best_epoch = 0
    print("Start training")
    start_time = time.time()
    Best_SR = 0

    result_list = []
    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
            train_stats = train(model, train_loader, optimizer, epoch, device, config) 

        #print('test on training set')
        #Train_Retrieve_SR = double_retrieval_eval_withrange(model, test_loader, 15, device, './output_coin/train_prediction.json' ,config['step_number'])

        print('test_result is:')
        Val_Retrieve_SR = double_retrieval_eval(model, val_loader, device, args.write_path,config['step_number']) 

        print('test_result is:')
        Test_Retrieve_SR = double_retrieval_eval(model, test_loader, device, args.write_path,config['step_number']) 
        
        if args.evaluate:
            break
        
        #record_current_result
        temp_dict = {}
        temp_dict['val_result'] = Val_Retrieve_SR
        temp_dict['test_result'] = Test_Retrieve_SR
        result_list.append(temp_dict)

        with open(args.result_path,'w') as f:
            json.dump(result_list,f)

        if Val_Retrieve_SR['overall_acc']>Best_SR:
            Best_SR = Test_Retrieve_SR['overall_acc']
            best_epoch = epoch
        
        # save object anyway
        save_obj = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': config,
            'epoch': epoch,
        }
        dataset_name = config['dataset']
        step_number = config['step_number']
        torch.save(save_obj, os.path.join(args.output_dir, f'checkpoint_{dataset_name}_epoch_{epoch}_T_{step_number}_retrieval.pth')) 

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='./configs/coin_double_2.yaml')
    parser.add_argument('--output_dir', default='./output/coin_double_2/text_retrieval_coin/')        
    parser.add_argument('--checkpoint', default='')   
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--write_path', default='./output/coin_double_2/test_prediction.json')
    parser.add_argument('--result_path', default='./output/coin_double_2/results.json')
    
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    #copy config file into the output directory.
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)