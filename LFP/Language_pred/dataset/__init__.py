import torch
from torch.utils.data import DataLoader

from dataset.coin_text_dataset import coin_text_dataset,coin_text_dataset_all,coin_text_dataset_all_213,coin_text_dataset_all_123,prompt_with_captioning_dataset

#create dataset
def create_dataset(dataset, config, mode):
    if dataset=='coin':          
        train_dataset = coin_text_dataset(config['train_file'],mode)
        val_dataset = coin_text_dataset(config['val_file'],mode)
        test_dataset = coin_text_dataset(config['test_file'],mode)                
        return train_dataset, val_dataset, test_dataset
    elif dataset == 'coin_all':
        train_dataset = coin_text_dataset_all(config['train_file'],mode)
        val_dataset = coin_text_dataset_all(config['val_file'],mode)
        test_dataset = coin_text_dataset_all(config['test_file'],mode)                
        return train_dataset, val_dataset, test_dataset
    elif dataset == 'coin_all_213':
        train_dataset = coin_text_dataset_all_213(config['train_file'],mode)
        val_dataset = coin_text_dataset_all_213(config['val_file'],mode)
        test_dataset = coin_text_dataset_all_213(config['test_file'],mode)                
        return train_dataset, val_dataset, test_dataset
    elif dataset == 'coin_all_123':
        train_dataset = coin_text_dataset_all_123(config['train_file'],mode)
        val_dataset = coin_text_dataset_all_123(config['val_file'],mode)
        test_dataset = coin_text_dataset_all_123(config['test_file'],mode)                
        return train_dataset, val_dataset, test_dataset
    elif dataset == 'captioning_all':
        train_dataset = prompt_with_captioning_dataset(config['train_file'],config['caption_train'],3)
        val_dataset = prompt_with_captioning_dataset(config['val_file'],config['caption_val'],3)
        test_dataset = prompt_with_captioning_dataset(config['test_file'],config['caption_test'],3)                
        return train_dataset, val_dataset, test_dataset

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