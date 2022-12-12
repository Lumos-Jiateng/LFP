import torch
from torch.utils.data import DataLoader
from data.dataset import Train_CrossTask_Retrieve_Dataset_with_Taskname_Double,Eval_CrossTask_Retrieve_Dataset_with_Taskname
from data.coin_dataset import Coin_Retrieve_Dataset_with_Taskname_Double,Crosstask_Retrieve_Dataset_with_Taskname_Double,Coin_Retrieve_Dataset_without_Taskname_Double,Coin_Retrieve_Dataset_with_Taskname_single_infer
from data.coin_multi_dataset import Coin_Retrieve_Dataset_with_Taskname_Multi

from torchvision import transforms
from transform.randaugment import RandomAugment
from torchvision.transforms.functional import InterpolationMode

#create dataset
def create_dataset(config):
    min_scale = 0.5
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    transform_train = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_size'],scale=(min_scale, 1.0),interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])        
    transform_test = transforms.Compose([
        transforms.Resize((config['image_size'],config['image_size']),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])
    
    '''
    if config['dataset']=='crosstask':          
        train_dataset = Train_CrossTask_Retrieve_Dataset_with_Taskname_Double(config['re_train_file'],config['cross_task_feature_path'])
        val_dataset = Eval_CrossTask_Retrieve_Dataset_with_Taskname(config['re_val_file'],config['cross_task_feature_path'])
        test_dataset = Eval_CrossTask_Retrieve_Dataset_with_Taskname(config['re_test_file'],config['cross_task_feature_path'])                
        return train_dataset, val_dataset, test_dataset
    '''
    if config['dataset'] == 'coin':
        train_dataset = Coin_Retrieve_Dataset_with_Taskname_Double(config['re_train_file'],transform_train)
        val_dataset = Coin_Retrieve_Dataset_with_Taskname_Double(config['re_val_file'],transform_test)
        test_dataset = Coin_Retrieve_Dataset_with_Taskname_Double(config['re_test_file'],transform_test)
        return train_dataset,val_dataset,test_dataset
    
    elif config['dataset'] == 'crosstask':
        train_dataset = Crosstask_Retrieve_Dataset_with_Taskname_Double(config['re_train_file'],transform_train)
        val_dataset = Crosstask_Retrieve_Dataset_with_Taskname_Double(config['re_val_file'],transform_test)
        test_dataset = Crosstask_Retrieve_Dataset_with_Taskname_Double(config['re_test_file'],transform_test)
        return train_dataset,val_dataset,test_dataset

    elif config['dataset'] == 'coin_no_name':
        train_dataset = Coin_Retrieve_Dataset_without_Taskname_Double(config['re_train_file'],transform_train)
        val_dataset = Coin_Retrieve_Dataset_without_Taskname_Double(config['re_val_file'],transform_test)
        test_dataset = Coin_Retrieve_Dataset_without_Taskname_Double(config['re_test_file'],transform_test)
        return train_dataset,val_dataset,test_dataset
    
    elif config['dataset'] == 'niv':
        train_dataset = Niv_Retrieve_Dataset_with_Taskname_Double(config['re_train_file'],transform_train)
        val_dataset = Niv_Retrieve_Dataset_with_Taskname_Double(config['re_val_file'],transform_test)
        test_dataset = Niv_Retrieve_Dataset_with_Taskname_Double(config['re_test_file'],transform_test)
        return train_dataset,val_dataset,test_dataset

    elif config['dataset'] == 'coin_multi':
        train_dataset = Coin_Retrieve_Dataset_with_Taskname_Multi(config['re_train_file'],transform_train)
        val_dataset = Coin_Retrieve_Dataset_with_Taskname_Multi(config['re_val_file'],transform_test)
        test_dataset = Coin_Retrieve_Dataset_with_Taskname_Multi(config['re_test_file'],transform_test)
        return train_dataset,val_dataset,test_dataset

    elif config['dataset'] == 'coin_single_infer':
        train_dataset = Coin_Retrieve_Dataset_with_Taskname_Double(config['re_train_file'],transform_train)
        val_dataset = Coin_Retrieve_Dataset_with_Taskname_single_infer(config['re_val_file'],transform_test)
        test_dataset = Coin_Retrieve_Dataset_with_Taskname_single_infer(config['re_test_file'],transform_test)
        return train_dataset,val_dataset,test_dataset

#create dataloader
def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    