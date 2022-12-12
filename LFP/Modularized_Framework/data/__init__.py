import torch
from torch.utils.data import DataLoader
from data.dataset import Coin_Infer_Train, Coin_Infer_Test,Image_Bart_Dataset,Image_Bart_Dataset_Predict_all,Image_Bart_Dataset_all,Image_Bart_Dataset_Predict_all_123
from data.dataset import Crosstask_Infer_Train,Crosstask_Infer_Test
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
    
    # dataset construction -- only depend on the train / test json file.
    # For training: Image can be single or multi, step number T = 3 / 4, data_usage can be alone/ALL
    # Head number can be 2 / 3 / 4 under T = 3 / 4 , now we test with 
    if config['dataset'] == 'transfer':
        if config['training_type'] == 'Multi_Image' and config['step_number'] == 'one':
            Train_Json_path = "/nfs4-p1/ljt/Code/BRE/niv_preprocessing/Json_for_transfer/Multi_Image/one_train.json"
            Val_Json_path = "/nfs4-p1/ljt/Code/BRE/niv_preprocessing/Json_for_transfer/Multi_Image/one_val.json"
            Test_Json_path = "/nfs4-p1/ljt/Code/BRE/niv_preprocessing/Json_for_transfer/Multi_Image/one_test.json"
        if config['training_type'] == 'Single_Image' and config['step_number'] == 'one':
            Train_Json_path = "/nfs4-p1/ljt/Code/BRE/niv_preprocessing/Json_for_transfer/Single_Image/one_train.json"
            Val_Json_path = "/nfs4-p1/ljt/Code/BRE/niv_preprocessing/Json_for_transfer/Single_Image/one_val.json"
            Test_Json_path = '/nfs4-p1/ljt/Code/BRE/niv_preprocessing/Json_for_transfer/Single_Image/one_test.json'
        if config['training_type'] == 'Multi_Image' and config['step_number'] == 'two':
            Train_Json_path = "/nfs4-p1/ljt/Code/BRE/niv_preprocessing/Json_for_transfer/Multi_Image/two_train.json"
            Val_Json_path = "/nfs4-p1/ljt/Code/BRE/niv_preprocessing/Json_for_transfer/Multi_Image/two_val.json"
            Test_Json_path = '/nfs4-p1/ljt/Code/BRE/niv_preprocessing/Json_for_transfer/Multi_Image/two_test.json'
        if config['training_type'] == 'Single_Image' and config['step_number'] == 'two':
            Train_Json_path = "/nfs4-p1/ljt/Code/BRE/niv_preprocessing/Json_for_transfer/Single_Image/two_train.json"
            Val_Json_path = "/nfs4-p1/ljt/Code/BRE/niv_preprocessing/Json_for_transfer/Single_Image/two_val.json"
            Test_Json_path = '/nfs4-p1/ljt/Code/BRE/niv_preprocessing/Json_for_transfer/Single_Image/two_test.json'
        train_dataset = Crosstask_Infer_Train(Train_Json_path,transform_train,config['head_number'])
        val_dataset = Crosstask_Infer_Test(Val_Json_path,transform_test)
        test_dataset = Crosstask_Infer_Test(Test_Json_path,transform_test)
        return train_dataset,val_dataset,test_dataset
    if config['dataset']=='coin' or config['dataset']=='coin_IB' or config['dataset'] == 'coin_IB_predict_all' or config['dataset'] == 'coin_IB_predict_all_123' or config['dataset'] == 'crosstask': 
        root_path = '/nfs4-p1/ljt/github/coin_preprocessing_new/Json_for_dataset_final/' 
        if config['dataset'] == 'crosstask':
            root_path = '/nfs4-p1/ljt/github/crosstask_preprocessing/Json_for_dataset_final/'
        assert config['strategy'] in ['alone','all']
        if config['strategy'] == 'alone':
            Train_Json_path = root_path + config['training_type'] + '/' + config['step_number'] + '_train.json'
        else:
            Train_Json_path = root_path + config['training_type'] + '/' + 'all' + '_train.json'
        Test_Json_path = root_path + config['testing_type'] + '/' + config['step_number'] + '_test.json'
            
        if config['dataset'] == 'coin':
            train_dataset = Coin_Infer_Train(Train_Json_path,transform_train,config['head_number'])
            val_dataset = Coin_Infer_Test(Test_Json_path,transform_test)
            test_dataset = Coin_Infer_Test(Test_Json_path,transform_test)
        elif config['dataset'] == 'crosstask':
            train_dataset = Crosstask_Infer_Train(Train_Json_path,transform_train,config['head_number'])
            val_dataset = Crosstask_Infer_Test(Test_Json_path,transform_test)
            test_dataset = Crosstask_Infer_Test(Test_Json_path,transform_test)
        else:
            if config['strategy'] == 'alone':
                if config['step_number'] == 'one':
                    Train_Json_path = root_path + config['training_type'] + '/' + 'IB_train_3.json'
                elif config['step_number'] == 'two':
                    Train_Json_path = root_path + config['training_type'] + '/' + 'IB_train_4.json'
                elif config['step_number'] == 'three':
                    Train_Json_path = root_path + config['training_type'] + '/' + 'IB_train_5.json'
                if config['dataset'] == 'coin_IB':
                    train_dataset = Image_Bart_Dataset(Train_Json_path,transform_train,config['head_number'])
                    val_dataset = Image_Bart_Dataset(Test_Json_path,transform_test,config['head_number'])
                    test_dataset = Image_Bart_Dataset(Test_Json_path,transform_test,config['head_number'])
                elif config['dataset'] == 'coin_IB_predict_all':
                    train_dataset = Image_Bart_Dataset_Predict_all(Train_Json_path,transform_train,config['head_number'])
                    val_dataset = Image_Bart_Dataset_Predict_all(Test_Json_path,transform_test,config['head_number'])
                    test_dataset = Image_Bart_Dataset_Predict_all(Test_Json_path,transform_test,config['head_number'])
                elif config['dataset'] == 'coin_IB_predict_all_123':
                    train_dataset = Image_Bart_Dataset_Predict_all_123(Train_Json_path,transform_train,config['head_number'])
                    val_dataset = Image_Bart_Dataset_Predict_all_123(Test_Json_path,transform_test,config['head_number'])
                    test_dataset = Image_Bart_Dataset_Predict_all_123(Test_Json_path,transform_test,config['head_number'])
            elif config['strategy'] == 'all':
                if config['step_number'] == 'one':
                    Train_Json_path = root_path + config['training_type'] + '/' + 'IB_train_3.json'
                elif config['step_number'] == 'two':
                    Train_Json_path = root_path + config['training_type'] + '/' + 'IB_train_4.json'
                train_dataset = Image_Bart_Dataset_all(Train_Json_path,transform_train,config['head_number'])
                val_dataset = Image_Bart_Dataset_all(Test_Json_path,transform_test,config['head_number'])
                test_dataset = Image_Bart_Dataset_all(Test_Json_path,transform_test,config['head_number'])
       
        return train_dataset,val_dataset,test_dataset
    
#create dataloader
def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = False#(sampler is None) 
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