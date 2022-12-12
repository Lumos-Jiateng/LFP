import os,sys
sys.path.append('/nfs4-p1/ljt/github/PTU/Double_retrieval')
from models.med import BertConfig
from models.nvlr_encoder import BertModel
from models.vit import interpolate_pos_embed
from models.blip import create_vit, init_tokenizer, is_url
from timm.models.hub import download_cached_file

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer
import numpy as np
import os
from data.crosstask_105 import step_list_105

language_sup_path = '/nfs4-p1/ljt/github/PTU/Full_coin_eval/language_sup/crosstask_full_features.npy'

class Coin_align(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 384,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 class_number = 105                   
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer, drop_path_rate=0.1)
        self.tokenizer = init_tokenizer()   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False) 
        text_width = self.text_encoder.config.hidden_size

        self.predict_visual_dim = 512          
        self.cls_head_for_ff = nn.Sequential(
                  nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
                  nn.ReLU(),
                  nn.Linear(self.text_encoder.config.hidden_size, class_number)
                )  
        self.cls_head_for_lf = nn.Sequential(
                  nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
                  nn.ReLU(),
                  nn.Linear(self.text_encoder.config.hidden_size, class_number)
                )    
        self.language_head_for_ff = nn.Sequential(
                  nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
                  nn.ReLU(),
                  nn.Linear(self.text_encoder.config.hidden_size, self.predict_visual_dim)
                ) 
        self.language_head_for_lf = nn.Sequential(
                  nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
                  nn.ReLU(),
                  nn.Linear(self.text_encoder.config.hidden_size, self.predict_visual_dim)
                )
                
        # needed if we want to use the same Bert to encoder language supervision
        self.text_projection_for_prediction = nn.Linear(text_width, self.predict_visual_dim) 

    def forward(self,text_description,ff_start,lf_end,ff_action_label,lf_action_label):
        
        #ff_start / lf_end are images that will be used.

        ff_start = self.visual_encoder(ff_start) 
        ff_start_atts = torch.ones(ff_start.size()[:-1],dtype=torch.long).to(ff_start.device) 
        lf_end = self.visual_encoder(lf_end) 
        lf_end_atts = torch.ones(lf_end.size()[:-1],dtype=torch.long).to(lf_end.device)

        text = self.tokenizer(text_description, padding='max_length', return_tensors="pt").to(ff_start.device) 
        output = self.text_encoder(text.input_ids, 
                                   attention_mask = text.attention_mask, 
                                   encoder_hidden_states = [ff_start,lf_end],
                                   encoder_attention_mask = [ff_start_atts,lf_end_atts],        
                                   return_dict = True,
                                  )  
        hidden_state = output.last_hidden_state[:,0,:]         
        ff_prediction = self.cls_head_for_ff(hidden_state)
        lf_prediction = self.cls_head_for_lf(hidden_state)
        ff_embedding = self.language_head_for_ff(hidden_state)
        lf_embedding = self.language_head_for_lf(hidden_state)
        
        ###============== classification loss ===================###

        #change to one-hot targets
        ff_target = F.one_hot(ff_action_label, num_classes=105) #[1,3] -> [1,0,0...],[0,0,1...]
        lf_target = F.one_hot(lf_action_label, num_classes=105) 

        loss_ff = F.cross_entropy(ff_prediction, ff_action_label)
        loss_lf = F.cross_entropy(lf_prediction, lf_action_label) # do not need one hot in cross-entropy

        ###============== language contrastive loss ===================###
        '''
        all_steps = self.tokenizer(step_list, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(ff_start.device)
        step_output = self.text_encoder(all_steps.input_ids, 
                                   attention_mask = all_steps.attention_mask,        
                                   return_dict = True,
                                   mode = 'text'
                                  )
        step_feat = F.normalize(self.text_projection_for_prediction(step_output.last_hidden_state[:,0,:]),dim=-1)
        '''
        step_feat = np.load(language_sup_path)
        step_feat = torch.tensor(step_feat).to(ff_start.device)
        
        ff_sim_matrix = ff_embedding @ step_feat.t()
        lf_sim_matrix = lf_embedding @ step_feat.t()
        language_loss_ff = -torch.sum(F.log_softmax(ff_sim_matrix, dim=1)*ff_target,dim=1).mean()
        language_loss_lf = -torch.sum(F.log_softmax(lf_sim_matrix, dim=1)*lf_target,dim=1).mean()
        
        return loss_ff , loss_lf , language_loss_ff , language_loss_lf
    
def load_crosstask_align_with_blip(pretrained='',**kwargs):
    model = Coin_align(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        print("missing keys:")
        print(msg.missing_keys)
        #print(msg.unexpected_keys)
    return model  

        
def load_checkpoint(model,url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu') 
    elif os.path.isfile(url_or_filename):        
        checkpoint = torch.load(url_or_filename, map_location='cpu') 
    else:
        raise RuntimeError('checkpoint url or path is invalid')
    state_dict = checkpoint['model']
    
    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder) 
    
    for key in list(state_dict.keys()):
        if 'crossattention.self.' in key:
            new_key0 = key.replace('self','self0')
            new_key1 = key.replace('self','self1')
            state_dict[new_key0] = state_dict[key]
            state_dict[new_key1] = state_dict[key]
        elif 'crossattention.output.dense.' in key:
            print('replacing cross attention')
            new_key0 = key.replace('dense','dense0')
            new_key1 = key.replace('dense','dense1')
            state_dict[new_key0] = state_dict[key]
            state_dict[new_key1] = state_dict[key]  
    for key in list(state_dict.keys()):
        print(key)            
    msg = model.load_state_dict(state_dict,strict=False)
    print('load checkpoint from %s'%url_or_filename)  
    return model,msg

if __name__ == '__main__':

    # try to make a load here
    url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth'
    model = load_crosstask_align_with_blip(pretrained = url)
