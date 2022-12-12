import argparse
import ruamel.yaml as yaml
from pathlib import Path
import os

from transformers import StoppingCriteria
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

from dataset import create_dataset,create_loader
from model.Bartbased_coin_finetune import text_baseline
from optim import create_optimizer
from scheduler import create_scheduler
import json

#from dataset.step_lists import Crosstask_primary_steps
#from model.constrained_decoding import MarisaTrie
#from model.constrained_decoding import cbs
from functools import partial 

def train(model, data_loader, optimizer, epoch, warmup_steps, device, scheduler, config):
    
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('text_loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps*step_size  

    for i,(b_Prompt,b_Predict) in enumerate(metric_logger.log_every(data_loader, print_freq, header)): 

        text_loss = model(b_Prompt,b_Predict,device)                  
        loss = text_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        metric_logger.update(text_loss=text_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)         
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()} 

def eval(model, data_loader, device, config, pred_path):

    model.eval()    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    start_time = time.time()
    #begin evaluation
    
    sample_number = 0         # sample number of dataset / procedure planning task number
    consistent_number = 0     # when predicted segment number is the same with sample segment number
    predict_number = 0        # Total count in segments
    correct_number = 0        # correct number for each segment
    success_number=0          # success number for sentences
    match_number=0            # intersection number
    set_cardinality = 0

    out_list = []
    count_number = 0

    for i,item in enumerate (data_loader):
        
        #Deal with a batch each time
        #print(item)
        Test=item[0]
        batch_size=len(item[0])
        Inputs=model.pretrained_tokenizer(Test, truncation=True, padding=True,return_tensors='pt')
        
        #beam_search used in generation
        summary_ids = model.pretrained_model.generate(Inputs['input_ids'].to(device,non_blocking=True), 
                                                      num_beams=8, max_length=50, early_stopping=True)
        #print example
        if i <= 10:
            print("Current Prompts are:")
            print(item[0][0])
            print("Predicted labels are:")
            print([model.pretrained_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0])
            print("Real predictions are:")
            print(item[1][0])
        
        for j in range(batch_size):
            temp_dict = {}
            real_pred = item[1][j]
            current_pred=[model.pretrained_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][j]
            current_pred_list=current_pred.split('.')
            real_pred_list=real_pred.split('.')
            
            #if(i==0 and j==0):
            print(real_pred_list,current_pred_list)
            if(current_pred_list[-1] == ''):
                current_pred_list.pop()
            if(real_pred_list[-1] == ''):
                real_pred_list.pop()
            
            mutual = 0
            #compute sample number
            sample_number=sample_number+1
            #compute consistent number
            if len(current_pred_list) == len(real_pred_list):
                consistent_number = consistent_number+1
            #compute predict number 
            predict_number = predict_number + len(real_pred_list)
            #compute correct_number
            for k in range(min(len(real_pred_list),len(current_pred_list))):
                if real_pred_list[k]==current_pred_list[k]:
                    correct_number=correct_number+1
                    mutual = mutual + 1
            #compute success number
            success_flag=1
            if len(current_pred_list) != len(real_pred_list):
                success_flag=0
            else:
                for t in range (len(current_pred_list)):
                    if real_pred_list[t] != current_pred_list[t]:
                        success_flag=0
            if success_flag == 1:
                success_number= success_number+1
            #compute match number:
            for s in range(len(current_pred_list)):
                current_pred = current_pred_list[s]  
                if current_pred in real_pred_list:
                    match_number=match_number+1
            set_cardinality = set_cardinality + len(current_pred_list) + len(real_pred_list) - mutual

            if success_flag == 1:
                temp_dict['number'] = count_number
                temp_dict['type'] = 'successful'
                temp_dict['target'] = real_pred_list
                temp_dict['predict'] = current_pred_list
                count_number = count_number + 1
            
            else:
                temp_dict['number'] = count_number
                temp_dict['type'] = 'Partly success'
                temp_dict['target'] = real_pred_list
                temp_dict['predict'] = current_pred_list
                count_number = count_number + 1
            
            out_list.append(temp_dict)

    print(f'sample number is {sample_number}')
    print(f'consistent_number is {consistent_number}')   
    print(f'predict_number is {predict_number}')
    print(f'correct_number is {correct_number}')
    print(f'success number is: {success_number}')

    print(f'Procedural-SR is {1.0*success_number/sample_number}')
    print(f'Step-SR is {1.0*correct_number/predict_number}')
    print(f'Step-acc is {1.0*match_number/predict_number}')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))

    with open(pred_path,'w') as f:
        json.dump(out_list,f)

    return {'Procedural-SR':1.0*success_number/sample_number,
            'Step-SR': 1.0*correct_number/predict_number,
            'Step-acc': 1.0*match_number/predict_number}

def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating coin_text dataset")
    train_dataset, val_dataset,test_dataset = create_dataset(config['dataset'], config ,args.prompt_mode)  
    
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

    print(train_dataset[0])

    #### Model #### 
    print("Creating model")
    model = text_baseline(config=config, pretrianed_model=args.pretrained_model)
    
    '''never need a checkpoint in this model'''
    if args.checkpoint:    
        checkpoint_path = args.checkpoint
        print(f'load_checkpoint_from{checkpoint_path}')
        ckpt = torch.load(checkpoint_path)
        state_dict = ckpt['model']
        msg = model.load_state_dict(state_dict,strict=False)
        print(msg)
           
    model = model.to(device)   
    
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)  
    
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    print("Start training")
    start_time = time.time()   
    Best_Procedural_SR = 0
    result = []
    for epoch in range(0, max_epoch):
        if not args.evaluate:
            train_stats = train(model, train_loader, optimizer, epoch, warmup_steps, device, lr_scheduler, config)  

        print('val_result is:')    
        Procedural_SR = eval(model, val_loader, device, config,args.pred_path)
        Test_SR = eval(model, test_loader, device, config,args.pred_path)

        if args.evaluate:
            break
        
        if Procedural_SR['Procedural-SR']>Best_Procedural_SR:
            Best_Procedural_SR = Procedural_SR['Procedural-SR']
            
        save_obj = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'config': config,
            'epoch': epoch,
            'Procedural-SR':Procedural_SR,
            'Test-SR':Test_SR
        }
        dataset_name = config['dataset']
        step_number = config['step_number']
        torch.save(save_obj, os.path.join(args.output_dir, f'checkpoint_{dataset_name}_epoch_{epoch}_mode_{args.prompt_mode}_T_{step_number}.pth')) 
        
        temp_dict = {}
        temp_dict['val_results'] = Procedural_SR
        temp_dict['test-results'] = Test_SR
        result.append(temp_dict)
        with open(args.result_path,'w') as f:
            json.dump(result,f)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='./configs/coin_text_step_4.yaml')
    parser.add_argument('--pretrained_model', default='BartForConditionalGerneration')
    parser.add_argument('--output_dir', default='./output/coin_text_step_4/text_baseline') 
    parser.add_argument('--result_path',default = './output/coin_text_step_4/Bart_result.json')
    parser.add_argument('--pred_path',default = './output/coin_text_step_4/pred_result.json')       
    parser.add_argument('--checkpoint', default='')   
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--prompt_mode',default=2,type=int)
    
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    #copy config file into the output directory.
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)